import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from copy import deepcopy
from contextlib import nullcontext
from datasets import Dataset
from torch.utils.data import ConcatDataset, DataLoader

from utils.metric import ResultSummary, ResultSummary2
from utils.backbone import get_backbone, obtain_features          # obtain_features 사용
from utils.optimizer import get_optimizer
from utils.dataloader import (
    get_dataloader,
    preprocess_function_train_generative_LAMOL,
)
from utils.evaluation import (
    evaluate_sent_level_acc_with_generation,
    evaluate_sent_level_acc_with_lmhead,
)
from models.Base import BaseLearner

logger = logging.getLogger()

# --------------------------------------------------------------------------
#                               파라미터
# --------------------------------------------------------------------------
def get_LAMOL_KD_params(parser):
    parser.add_argument("--LAMOL_KD_distill_weight", type=float, default=1.0)
    parser.add_argument("--LAMOL_KD_lambda", type=float, default=0.25)
    parser.add_argument("--LAMOL_KD_gamma", type=float, default=0.20)
    parser.add_argument("--LAMOL_KD_topk", type=int, default=20)
    parser.add_argument("--LAMOL_KD_temperature", type=float, default=2.0)
    parser.add_argument("--LAMOL_use_task_specific_gen_token", type=bool, default=True)

# ==========================================================================
#                               LAMOL_KD
# ==========================================================================
class LAMOL_KD(BaseLearner):
    """
    • 학습(각 task)  : 기존 LAMOL_KD 방식 — backbone + LM-Head 함께 업데이트
    • 평가(Dev/Test) : backbone 고정 → LM-Head만 짧게 미세조정 → 정확도 측정 후 원복
    """

    # ------------------------------ init ------------------------------
    def __init__(self, params, CL_dataset, accelerator):
        super().__init__(params, CL_dataset, accelerator)

        assert params.classifier == "None"
        assert params.il_mode in ["IIL", "CIL", "TIL"]
        assert params.classification_type == "sentence-level"
        assert params.backbone_type == "generative"
        assert not params.is_replay

    # =========================== build 단계 ===========================
    def build_metric(self):
        cfg = self.CL_dataset.continual_config
        self.result_summary                = ResultSummary(cfg["NUM_TASK"])
        self.probing_result_summary        = ResultSummary(cfg["NUM_TASK"])
        self.kmeans_result_summary         = ResultSummary2(cfg["NUM_TASK"])
        self.gmm_result_summary            = ResultSummary2(cfg["NUM_TASK"])
        self.spectral_result_summary       = ResultSummary2(cfg["NUM_TASK"])
        self.deep_clustering_result_summary= ResultSummary2(cfg["NUM_TASK"])
        self.agglomerative_result_summary  = ResultSummary2(cfg["NUM_TASK"])
        if self.params.il_mode == "IIL":
            self.result_summary_train = ResultSummary(cfg["NUM_TASK"])

    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def build_classifier(self):
        self.classifier_list = None

    def build_optimizer(self):
        # backbone + LM-Head 통합 옵티마이저
        self.optimizer = get_optimizer(self.params, self.model, None)

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = \
            get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None   # LAMOL_KD 는 replay 미사용

    def accelerate_prepare(self):
        prep = self.accelerator.prepare
        self.model, self.optimizer, *self.train_loader_list = \
            prep(self.model, self.optimizer, *self.train_loader_list)
        self.dev_loader_list  = prep(*self.dev_loader_list)
        self.test_loader_list = prep(*self.test_loader_list)
        self.teacher_model = None

    # ────────────────────── Helper ─ teacher 해제 ─────────────────────
    def _release_teacher(self):
        if getattr(self, "teacher_model", None) is not None:
            # accelerate hooks가 있는 모델은 .to() 사용하지 않음
            del self.teacher_model
            self.teacher_model = None
            torch.cuda.empty_cache()

    # ========================= Task hooks =========================
    def begin_task(self, task_id):
        super().begin_task(task_id)
        self._release_teacher()

        if task_id == 0:
            return

        # teacher self-distillation - 메모리 효율적 방식
        from utils.backbone import get_backbone
        import gc
        
        # 1) 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
        
        # 2) teacher 모델을 처음부터 새로 로드 (device_map 문제 회피)
        self.teacher_model, _ = get_backbone(self.params)
        
        # 3) student의 state_dict를 teacher에 복사 (메모리 효율적)
        student_model = self.model.module if hasattr(self.model, "module") else self.model
        
        # state_dict 복사 시 메모리 효율화
        with torch.no_grad():
            for (s_name, s_param), (t_name, t_param) in zip(
                student_model.named_parameters(), 
                self.teacher_model.named_parameters()
            ):
                if s_name == t_name:  # 파라미터 이름 매칭
                    t_param.copy_(s_param.cpu())
        
        # 4) teacher 최적화 설정 (gradient checkpointing 활용)
        if hasattr(self.teacher_model, 'gradient_checkpointing_enable'):
            self.teacher_model.gradient_checkpointing_enable()
            
        teacher_opt = get_optimizer(self.params, self.teacher_model, None)
        teacher_loader = self.train_loader_list[task_id]
        
        # 5) teacher는 CPU에서 학습 (메모리 절약)
        teacher_opt = get_optimizer(self.params, self.teacher_model, None)
        t_model = self.teacher_model
        t_model.train()

        # 6) teacher 학습 (CPU 기반 - 메모리 효율적)
        for epoch in range(self.params.training_epochs):
            for lm_input in teacher_loader:
                # 입력을 CPU로 이동
                cpu_input = {}
                for k, v in lm_input.items():
                    if isinstance(v, torch.Tensor):
                        cpu_input[k] = v.cpu()
                    else:
                        cpu_input[k] = v
                
                # CPU에서 teacher 학습
                qa  = t_model(
                    input_ids=cpu_input["input_ids_with_ans"],
                    attention_mask=cpu_input["attention_mask_with_ans"],
                    labels=cpu_input["labels_with_ans"],
                ).loss
                gen = t_model(
                    input_ids=cpu_input["input_ids_with_gen_ans"],
                    attention_mask=cpu_input["attention_mask_with_gen_ans"],
                    labels=cpu_input["labels_with_gen_ans"],
                ).loss
                loss = qa + self.params.LAMOL_KD_lambda * gen
                
                teacher_opt.zero_grad()
                loss.backward()
                teacher_opt.step()
                
                # 배치마다 메모리 정리
                del qa, gen, loss, cpu_input
                torch.cuda.empty_cache()

    def end_task(self, task_id):
        self._release_teacher()
        super().end_task(task_id)

    # ========================= Epoch loop =========================
    def train_epochs(self, task_id):
        """KD + backbone+LM-Head 동시 업데이트(기존 방식 유지)"""
        if task_id > 0:
            base_ds = self.train_loader_list[task_id].dataset
            pseudo  = self.generate_pseudo_buffer_samples(
                task_id, int(len(base_ds) * self.params.LAMOL_KD_gamma)
            )
            # device_map 유지를 위해 새로운 loader 생성 시 주의
            combined_ds = ConcatDataset((base_ds, *pseudo))
            loader = DataLoader(
                combined_ds,
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False,
            )
            # 동적 loader도 accelerator prepare
            loader = self.accelerator.prepare(loader)
        else:
            loader = self.train_loader_list[task_id]

        for epoch in range(self.params.training_epochs):
            if self.accelerator.is_main_process:
                logger.info(f"------ [Train] epoch {epoch+1}/{self.params.training_epochs} ------")
            self.loss_list = []
            self.model.train()

            for lm_input in loader:
                self.observe_batch(task_id, epoch, lm_input)

            if self.loss_list:
                mean_loss = np.mean(self.loss_list)
                if self.accelerator.is_main_process:
                    logger.info(f"[Train] Epoch {epoch+1}: mean_loss={mean_loss:.3f}")
                self.accelerator.log({"train_loss": mean_loss}, step=self.global_step)

            if self.params.evaluate_interval > 0 and epoch % self.params.evaluate_interval == 0:
                acc, _ = self.evaluate_current_task(task_id, task_id, "dev", self.params.il_mode)
                if self.accelerator.is_main_process:
                    logger.info(f"[Dev] Acc={acc:.3f}")
                self.accelerator.log({f"Dev_Acc_Task_{task_id}": acc}, step=self.global_step)

            torch.cuda.empty_cache()

        self._release_teacher()

    # ---------------------- observe_batch ----------------------
    def observe_batch(self, task_id, epoch_id, lm_input):
        self.step        += 1
        self.global_step += 1

        student = self.model.module if hasattr(self.model, "module") else self.model
        teacher = (
            self.teacher_model.module if (self.teacher_model and hasattr(self.teacher_model, "module")) else self.teacher_model
        )

        ids_ans, mask_ans, lbls_ans = lm_input["input_ids_with_ans"], lm_input["attention_mask_with_ans"], lm_input["labels_with_ans"]
        ids_gen, mask_gen, lbls_gen = lm_input["input_ids_with_gen_ans"], lm_input["attention_mask_with_gen_ans"], lm_input["labels_with_gen_ans"]
        idx_cil = lm_input["label_idx_cil"]

        # student의 주 device 찾기 (multi-GPU 환경 고려)
        student_device = next(student.parameters()).device
        
        if task_id == 0:
            qa  = student(input_ids=ids_ans, attention_mask=mask_ans, labels=lbls_ans).loss
            gen = student(input_ids=ids_gen, attention_mask=mask_gen, labels=lbls_gen).loss
            loss = qa + self.params.LAMOL_KD_lambda * gen
        else:
            cur = idx_cil != -1
            buf = idx_cil == -1

            # 모든 loss tensor를 동일한 device로 초기화
            loss_cur = torch.tensor(0., device=student_device)
            if cur.any():
                qa_c  = student(input_ids=ids_ans[cur], attention_mask=mask_ans[cur], labels=lbls_ans[cur]).loss
                gen_c = student(input_ids=ids_gen[cur], attention_mask=mask_gen[cur], labels=lbls_gen[cur]).loss
                loss_cur = (qa_c + self.params.LAMOL_KD_lambda * gen_c).to(student_device, non_blocking=True)

            loss_buf = torch.tensor(0., device=student_device)
            if buf.any() and teacher is not None:
                # CPU에서 teacher inference (메모리 극대 절약)
                cpu_ids_ans = ids_ans[buf].cpu()
                cpu_mask_ans = mask_ans[buf].cpu()
                cpu_lbls_ans = lbls_ans[buf].cpu()
                cpu_ids_gen = ids_gen[buf].cpu()
                cpu_mask_gen = mask_gen[buf].cpu()
                cpu_lbls_gen = lbls_gen[buf].cpu()
                
                with torch.no_grad():
                    t_qa  = teacher(input_ids=cpu_ids_ans, attention_mask=cpu_mask_ans, labels=cpu_lbls_ans).logits
                    t_gen = teacher(input_ids=cpu_ids_gen, attention_mask=cpu_mask_gen, labels=cpu_lbls_gen).logits
                
                # 마스킹 후 GPU로 이동
                keep_qa  = cpu_lbls_ans != -100
                keep_gen = cpu_lbls_gen != -100
                t_qa = t_qa[keep_qa].to(student_device, non_blocking=True)
                t_gen = t_gen[keep_gen].to(student_device, non_blocking=True)

                # student inference (GPU)
                with torch.cuda.amp.autocast(enabled=self.accelerator.mixed_precision in ("fp16", "bf16")):
                    s_qa_full = student(input_ids=ids_ans[buf], attention_mask=mask_ans[buf], labels=lbls_ans[buf]).logits
                    s_gen_full= student(input_ids=ids_gen[buf], attention_mask=mask_gen[buf], labels=lbls_gen[buf]).logits
                
                # logits의 실제 device 확인 후 keep mask를 같은 device로 이동
                logits_device = s_qa_full.device
                keep_qa_gpu = keep_qa.to(logits_device, non_blocking=True)
                keep_gen_gpu = keep_gen.to(logits_device, non_blocking=True)
                
                # 안전한 인덱싱 후 student_device로 이동
                s_qa = s_qa_full[keep_qa_gpu]
                s_gen= s_gen_full[keep_gen_gpu]
                
                # 최종적으로 student_device로 이동 (필요시)
                if s_qa.device != student_device:
                    s_qa = s_qa.to(student_device, non_blocking=True)
                if s_gen.device != student_device:
                    s_gen = s_gen.to(student_device, non_blocking=True)
                
                # CPU 텐서들 정리
                del cpu_ids_ans, cpu_mask_ans, cpu_lbls_ans, cpu_ids_gen, cpu_mask_gen, cpu_lbls_gen
                del s_qa_full, s_gen_full

                # KL loss 계산
                self.kl_loss = self.kl_loss.to(student_device)
                t = self.params.LAMOL_KD_temperature
                kd_qa  = self.kl_loss(F.log_softmax(s_qa / t, dim=-1), F.softmax(t_qa / t, dim=-1)) * t**2
                kd_gen = self.kl_loss(F.log_softmax(s_gen/ t, dim=-1), F.softmax(t_gen/ t, dim=-1)) * t**2
                loss_buf = (kd_qa + self.params.LAMOL_KD_lambda * kd_gen).to(student_device, non_blocking=True)
                
                # 모든 중간 텐서들 정리
                del t_qa, t_gen, s_qa, s_gen, kd_qa, kd_gen, keep_qa, keep_gen, keep_qa_gpu, keep_gen_gpu, logits_device
                torch.cuda.empty_cache()

            # 두 loss 모두 student_device에 있는지 확인
            loss_cur = loss_cur.to(student_device, non_blocking=True) 
            loss_buf = loss_buf.to(student_device, non_blocking=True)
            loss = loss_cur + self.params.LAMOL_KD_distill_weight * loss_buf

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            self.optimizer.step()
            self.loss_list.append(loss.item())

        if self.params.info_per_steps and self.step % self.params.info_per_steps == 0 and self.loss_list:
            mean_loss = np.mean(self.loss_list)
            if self.accelerator.is_main_process:
                logger.info(f"[Train] Epoch {epoch_id+1}, Step {self.step}: mean_loss={mean_loss:.3f}")
            self.accelerator.log({"train_loss": mean_loss}, step=self.global_step)

    # ======================== Evaluation ========================
    def evaluate_current_task(self, eval_task_id, cur_task_id, phase, il_mode):
        loaders = (
            self.train_loader_list if phase == "train" else
            self.dev_loader_list   if phase == "dev"   else
            self.test_loader_list
        )
        next_loader = loaders[eval_task_id + 1] if eval_task_id + 1 < len(loaders) else None
        model = self.model.module if hasattr(self.model, "module") else self.model

        # ───── 평가용 LM-Head mini-finetune ─────
        if self.params.lm_head_finetune:
            lm_head = model.get_output_embeddings()
            orig_state = deepcopy(lm_head.state_dict())

            for p in model.parameters():  # backbone freeze
                p.requires_grad = False
            for p in lm_head.parameters():
                p.requires_grad = True

            ft_loader = self.train_loader_list[eval_task_id]
            ft_opt = torch.optim.AdamW(lm_head.parameters(), lr=self.params.lm_head_lr)

            autocast_ctx = (
                torch.cuda.amp.autocast
                if self.accelerator.mixed_precision in ("fp16", "bf16")
                else nullcontext
            )
            model.eval()
            for _ in range(self.params.lm_head_epochs):
                for lm_input in ft_loader:
                    with torch.no_grad(), autocast_ctx():
                        feats = obtain_features(self.params, model, lm_input, self.tokenizer)
                    dev = lm_head.weight.device
                    if feats.device != dev:
                        feats = feats.to(dev, non_blocking=True)

                    if "target" in lm_input:
                        enc = self.tokenizer(lm_input["target"], add_special_tokens=False, padding=False)
                        label_tok = torch.tensor([ids[0] for ids in enc["input_ids"]], device=dev)
                    else:
                        first = []
                        for row in lm_input["labels_with_ans"]:
                            nz = (row != -100).nonzero(as_tuple=False)
                            first.append(row[nz[0]].item() if nz.numel() else self.tokenizer.eos_token_id)
                        label_tok = torch.tensor(first, device=dev)

                    logits = lm_head(feats)
                    loss = F.cross_entropy(logits, label_tok)
                    ft_opt.zero_grad()
                    loss.backward()
                    ft_opt.step()

                    del feats, logits, label_tok, loss
                    torch.cuda.empty_cache()

            acc_cur, acc_next = evaluate_sent_level_acc_with_lmhead(
                model=model,
                cur_task_id=cur_task_id,
                eval_data_loader=loaders[eval_task_id],
                next_eval_data_loader=next_loader,
                tokenizer=self.tokenizer,
                accelerator=self.accelerator,
                params=self.params,
                idx2label=self.CL_dataset.continual_config["idx2label"],
            )

            lm_head.load_state_dict(orig_state)  # 원복
            for p in model.parameters():
                p.requires_grad = True
            torch.cuda.empty_cache()
            return acc_cur, acc_next

        # ───── 기본 generation 평가 ─────
        acc_cur, acc_next = evaluate_sent_level_acc_with_generation(
            model=model,
            eval_data_loader=loaders[eval_task_id],
            next_eval_data_loader=next_loader,
            tokenizer=self.tokenizer,
            accelerator=self.accelerator,
            params=self.params,
            idx2label=self.CL_dataset.continual_config["idx2label"],
        )
        return acc_cur, acc_next

    # ===================================================================
    #          generate_pseudo_buffer_samples (원본 코드 유지)
    # ===================================================================
    def generate_pseudo_buffer_samples(self, task_id: int, num_samples: int) -> List[Dataset]:
        """
        원본 LAMOL_KD 방식과 동일
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        input_column, target_column = "input", "target"
        ans_token, eos_token = "__ans__", self.tokenizer.eos_token
        num_task = self.CL_dataset.continual_config["NUM_TASK"]
        pseudo_dataset_list: List[Dataset] = []
        generate_batch_size = 8

        with torch.no_grad():
            for t_id in range(task_id):
                sample_dict = {
                    "input": [], "target": [], "label_idx_cil": [], "label_idx_til": []
                }
                if self.params.il_mode == "IIL":
                    sample_dict.update({"instance_id": [], "concept_id": [], "relation_id": []})

                cnt = num_samples // task_id
                gen_token = f"__{t_id}__" if self.params.LAMOL_use_task_specific_gen_token else "__gen__"

                while cnt > 0:
                    n = min(generate_batch_size, cnt)
                    lm_in = self.tokenizer([gen_token]*n, padding="longest", return_tensors="pt")
                    lm_in = {k: v.to(model.device) for k, v in lm_in.items()}
                    max_len = max(len(ids) for ids in lm_in["input_ids"])
                    gen_ids_all = model.generate(
                        **lm_in,
                        max_new_tokens=self.params.max_seq_length - max_len,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=self.params.LAMOL_KD_topk,
                    )
                    gen_ids = gen_ids_all[:, max_len:]
                    gen_txt = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=False)

                    for txt in gen_txt:
                        if txt.count(ans_token) != 1:
                            continue
                        q, a = txt.split(ans_token)
                        a = a.replace(self.tokenizer.eos_token, "")
                        sample_dict["input"].append(q)
                        sample_dict["target"].append(a)
                        sample_dict["label_idx_cil"].append(-1)
                        sample_dict["label_idx_til"].append(-1)
                        if self.params.il_mode == "IIL":
                            sample_dict["instance_id"].append(-1)
                            sample_dict["concept_id"].append(-1)
                            sample_dict["relation_id"].append(-1)
                    cnt -= n

                if not sample_dict["input"]:
                    logger.info(f"No pseudo samples generated for task {t_id+1}!")
                    continue

                pseudo_ds = Dataset.from_dict(sample_dict)
                pseudo_ds = pseudo_ds.map(
                    preprocess_function_train_generative_LAMOL,
                    batched=True,
                    desc=f"Generate pseudo samples for task {t_id+1}",
                    batch_size=1000,
                    fn_kwargs={
                        "params": self.params,
                        "tokenizer": self.tokenizer,
                        "num_task": num_task,
                        "task_id": t_id,
                        "input_column": input_column,
                        "target_column": target_column,
                        "ans_token": ans_token,
                        "eos_token": eos_token,
                        "gen_token": gen_token,
                    },
                )
                cols = [
                    "input_ids", "attention_mask", "label_idx_cil", "label_idx_til",
                    "input_ids_with_ans", "attention_mask_with_ans", "labels_with_ans",
                    "input_ids_with_gen_ans", "attention_mask_with_gen_ans", "labels_with_gen_ans",
                ]
                if self.params.il_mode == "IIL":
                    cols += ["target", "instance_id", "concept_id", "relation_id"]
                pseudo_ds.set_format(type="torch", columns=cols)
                pseudo_dataset_list.append(pseudo_ds)

        return pseudo_dataset_list