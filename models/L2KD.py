import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from copy import deepcopy
from datasets import Dataset
from torch.utils.data import ConcatDataset, DataLoader
from contextlib import nullcontext

from utils.metric import ResultSummary, ResultSummary2
from utils.backbone import get_backbone, obtain_features
from utils.optimizer import get_optimizer
from utils.dataloader import (
    get_dataloader,
    preprocess_function_train_generative_LAMOL,
)
from utils.buffer import get_buffer  # (not used, but kept for future)
from utils.evaluation import (
    evaluate_sent_level_acc_with_generation,
    evaluate_sent_level_acc_with_classifier,
    evaluate_sent_level_acc_with_lmhead,
)
from models.Base import BaseLearner

logger = logging.getLogger()

# ------------------------------------------------------------------------------
#                                파라미터
# ------------------------------------------------------------------------------

def get_L2KD_params(parser):
    parser.add_argument("--L2KD_distill_weight", type=float, default=1.0)
    parser.add_argument("--L2KD_lambda", type=float, default=0.25)
    parser.add_argument("--L2KD_gamma", type=float, default=0.20)
    parser.add_argument("--L2KD_topk", type=int, default=20)
    parser.add_argument("--L2KD_temperature", type=float, default=2.0)
    parser.add_argument("--LAMOL_use_task_specific_gen_token", type=bool, default=True)
    # lm_head_finetune, lm_head_lr, lm_head_epochs 는 상위 파서(SEQ 등)에 이미 정의되어 있다고 가정

# ==============================================================================
#                         L2KD  (two-stage version)
# ==============================================================================
class L2KD(BaseLearner):
    """
    L2KD with optional two-stage training:
        1) Stage-1 (default): Train backbone with KD loss.
        2) Stage-2 (optional, --lm_head_finetune):
           Freeze backbone ▸ fine-tune LM-Head only with CE loss.
    """

    # -------------------------- init 제약 --------------------------
    def __init__(self, params, CL_dataset, accelerator):
        super().__init__(params, CL_dataset, accelerator)

        assert params.classifier == "None"
        assert params.il_mode in ["IIL", "CIL", "TIL"]
        assert params.classification_type == "sentence-level"
        assert params.backbone_type == "generative"
        assert not params.is_replay

        # two-stage flags (updated inside train_epochs)
        self.backbone_update: bool = True
        self.lm_head_update: bool = False

    # =========================== build 단계 =========================
    def build_metric(self):
        self.result_summary                = ResultSummary(self.CL_dataset.continual_config["NUM_TASK"])
        self.probing_result_summary        = ResultSummary(self.CL_dataset.continual_config["NUM_TASK"])
        self.kmeans_result_summary         = ResultSummary2(self.CL_dataset.continual_config["NUM_TASK"])
        self.gmm_result_summary            = ResultSummary2(self.CL_dataset.continual_config["NUM_TASK"])
        self.spectral_result_summary       = ResultSummary2(self.CL_dataset.continual_config["NUM_TASK"])
        self.deep_clustering_result_summary= ResultSummary2(self.CL_dataset.continual_config["NUM_TASK"])
        self.agglomerative_result_summary  = ResultSummary2(self.CL_dataset.continual_config["NUM_TASK"])
        if self.params.il_mode == "IIL":
            self.result_summary_train = ResultSummary(self.CL_dataset.continual_config["NUM_TASK"])

    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def build_classifier(self):
        self.classifier_list = None  # L2KD 는 분류기 사용 안 함

    def build_optimizer(self):
        # (1) backbone optimizer
        self.optimizer = get_optimizer(self.params, self.model, None)

        # (2) LM-Head 전용 optimizer (초기에 placeholder, Stage-2에서 재생성)
        self.lm_head_optimizer = None
        if getattr(self.params, "lm_head_finetune", False):
            lm_head = self.model.get_output_embeddings()
            self.lm_head_optimizer = torch.optim.AdamW([lm_head.weight], lr=self.params.lm_head_lr)

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(
            self.params, self.CL_dataset, self.tokenizer
        )

    def build_buffer(self):
        self.buffer = None

    def accelerate_prepare(self):
        # prepare backbone optimizer only
        self.model, self.optimizer, *self.train_loader_list = self.accelerator.prepare(
            self.model, self.optimizer, *self.train_loader_list
        )
        self.dev_loader_list  = self.accelerator.prepare(*self.dev_loader_list)
        self.test_loader_list = self.accelerator.prepare(*self.test_loader_list)

        # manually move LM-Head params (single tensor) to device
        if self.lm_head_optimizer is not None:
            for g in self.lm_head_optimizer.param_groups:
                g["params"] = [p.to(self.model.device) for p in g["params"]]

        self.teacher_model = None

    # ------------------------------------------------------------------
    #  Task-level hooks
    # ------------------------------------------------------------------
    def begin_task(self, task_id):
        super().begin_task(task_id)

        # ──────────────────────────────────────────────────────────────
        # 0) 기존 teacher 정리
        # ──────────────────────────────────────────────────────────────
        if getattr(self, "teacher_model", None) is not None:
            try:
                del self.teacher_model
                self.teacher_model = None
            except:
                pass
            torch.cuda.empty_cache()

        # ──────────────────────────────────────────────────────────────
        # 1) 첫 task 는 teacher 불필요 → 조기 반환
        # ──────────────────────────────────────────────────────────────
        if task_id == 0:
            return

        # ------------------------------------------------------------------
        # 2) **학생 모델**에 이번 task용 특수 토큰 삽입 & 임베딩 확장
        # ------------------------------------------------------------------
        new_token = f"__{task_id}__"
        if new_token not in self.tokenizer.get_vocab():
            added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [new_token]}
            )
            if added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.config.vocab_size = len(self.tokenizer)
                if self.accelerator.is_main_process:
                    logger.info(f"[Student] resize_token_embeddings → {len(self.tokenizer)}")

        # ------------------------------------------------------------------
        # 3) teacher 모델 로드 (CPU에서 생성하여 메모리 문제 방지)
        # ------------------------------------------------------------------
        try:
            # Memory cleanup before teacher creation
            torch.cuda.empty_cache()
            import gc
            gc.collect()

            if self.accelerator.is_main_process:
                logger.info(f"[Teacher] Creating teacher model for task {task_id} on CPU")

            # Create teacher model and move to CPU to avoid CUDA memory issues
            from utils.backbone import get_backbone
            
            # Create teacher model normally, then move to CPU
            self.teacher_model, _ = get_backbone(self.params, num_task=task_id + 1)
            
            # Move teacher model to CPU to save GPU memory
            self.teacher_model = self.teacher_model.cpu()
            
            if self.accelerator.is_main_process:
                logger.info(f"[Teacher] Teacher model moved to CPU")
            
            # ------------------------------------------------------------------
            # 4) student → teacher 파라미터 복사 (안전한 CPU 기반 복사)
            # ------------------------------------------------------------------
            if self.accelerator.is_main_process:
                logger.info(f"[Teacher] Copying parameters from student to teacher")
                
            student_model = self.model.module if hasattr(self.model, "module") else self.model
            
            with torch.no_grad():
                for (s_name, s_param), (t_name, t_param) in zip(
                    student_model.named_parameters(), self.teacher_model.named_parameters()
                ):
                    if s_name != t_name:
                        continue
                        
                    # Move student param to CPU for safe copying
                    s_param_cpu = s_param.detach().cpu()
                    
                    if s_param_cpu.shape == t_param.shape:
                        t_param.data.copy_(s_param_cpu)
                    else:
                        # 임베딩처럼 vocab 축만 다른 경우: 공통 부분만 복사
                        if (
                            s_param_cpu.dim() == t_param.dim() == 2
                            and s_param_cpu.shape[1] == t_param.shape[1]
                        ):
                            common_rows = min(s_param_cpu.shape[0], t_param.shape[0])
                            t_param.data[:common_rows].copy_(s_param_cpu[:common_rows])
                        else:
                            if self.accelerator.is_main_process:
                                logger.warning(
                                    f"[Teacher-copy] skip {t_name} due to shape mismatch "
                                    f"{s_param_cpu.shape} → {t_param.shape}"
                                )

            # teacher model 설정
            if hasattr(self.teacher_model, "config"):
                self.teacher_model.config.use_cache = False
            if hasattr(self.teacher_model, "gradient_checkpointing_disable"):
                self.teacher_model.gradient_checkpointing_disable()

            # ------------------------------------------------------------------
            # 5) teacher optimizer & dataloader (teacher는 CPU 학습)
            # ------------------------------------------------------------------
            if self.accelerator.is_main_process:
                logger.info(f"[Teacher] Starting self-distillation for task {task_id}")
                
            teacher_opt = get_optimizer(self.params, self.teacher_model, None)
            teacher_loader = self.train_loader_list[task_id]
            self.teacher_model.train()

            # ------------------------------------------------------------------
            # 6) teacher self-distillation (CPU 기반 안전 학습)
            # ------------------------------------------------------------------            
            for epoch in range(self.params.training_epochs):
                epoch_losses = []
                for batch_idx, lm_input in enumerate(teacher_loader):
                    try:
                        # 모든 입력을 CPU로 이동
                        cpu_input = {}
                        for k, v in lm_input.items():
                            if isinstance(v, torch.Tensor):
                                cpu_input[k] = v.cpu()
                            else:
                                cpu_input[k] = v

                        # Validate input IDs for teacher model
                        vocab_size = self.teacher_model.config.vocab_size
                        for key in ["input_ids_with_ans", "input_ids_with_gen_ans"]:
                            if key in cpu_input and isinstance(cpu_input[key], torch.Tensor):
                                if cpu_input[key].max() >= vocab_size or cpu_input[key].min() < 0:
                                    if self.accelerator.is_main_process:
                                        logger.warning(f"[Teacher] {key} contains invalid token IDs. Clamping to valid range.")
                                    cpu_input[key] = torch.clamp(cpu_input[key], 0, vocab_size - 1)

                        # Forward pass
                        qa_output = self.teacher_model(
                            input_ids=cpu_input["input_ids_with_ans"],
                            attention_mask=cpu_input["attention_mask_with_ans"],
                            labels=cpu_input["labels_with_ans"],
                        )
                        gen_output = self.teacher_model(
                            input_ids=cpu_input["input_ids_with_gen_ans"],
                            attention_mask=cpu_input["attention_mask_with_gen_ans"],
                            labels=cpu_input["labels_with_gen_ans"],
                        )
                        
                        qa_loss = qa_output.loss
                        gen_loss = gen_output.loss
                        
                        # Check for valid losses
                        if torch.isnan(qa_loss) or torch.isnan(gen_loss):
                            if self.accelerator.is_main_process:
                                logger.warning(f"[Teacher] NaN loss detected: qa_loss={qa_loss}, gen_loss={gen_loss}")
                            continue
                            
                        loss = qa_loss + self.params.L2KD_lambda * gen_loss
                        epoch_losses.append(loss.item())

                        # Backward pass
                        teacher_opt.zero_grad()
                        loss.backward()
                        teacher_opt.step()

                        # Clean up
                        del qa_output, gen_output, qa_loss, gen_loss, loss, cpu_input
                        
                    except Exception as e:
                        if self.accelerator.is_main_process:
                            logger.warning(f"[Teacher] Error in batch {batch_idx}: {str(e)}")
                        continue
                
                if epoch_losses and self.accelerator.is_main_process:
                    avg_loss = np.mean(epoch_losses)
                    logger.info(f"[Teacher] Epoch {epoch+1}/{self.params.training_epochs}, avg_loss={avg_loss:.4f}")

        except Exception as e:
            if self.accelerator.is_main_process:
                logger.error(f"[Teacher] Failed to create or train teacher model: {str(e)}")
                logger.warning(f"[Teacher] Proceeding without teacher model for task {task_id}")
            
            # Clean up any partial teacher model
            if hasattr(self, 'teacher_model'):
                try:
                    del self.teacher_model
                    self.teacher_model = None
                except:
                    pass
            
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def end_task(self, task_id):
        super().end_task(task_id)

    # ======================= Epoch-Level ===========================
    def train_epochs(self, task_id):
        """ Two-stage training loop """
        # ---------------- dataloader 준비 ----------------
        if task_id > 0:
            base_ds   = self.train_loader_list[task_id].dataset
            pseudo_ds = self.generate_pseudo_buffer_samples(task_id, int(len(base_ds) * self.params.L2KD_gamma))
            loader    = DataLoader(
                ConcatDataset((base_ds, *pseudo_ds)),
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False,
            )
            loader = self.accelerator.prepare(loader)
        else:
            loader = self.train_loader_list[task_id]

        # ==============================================================
        # Stage-1 ▸ Backbone + LM-Head 동시 학습 (LAMOL_KD와 동일)
        # ==============================================================
        self.backbone_update = True
        self.lm_head_update  = False

        for epoch_id in range(self.params.training_epochs):
            if self.accelerator.is_main_process:
                logger.info(f"------ [Train] epoch {epoch_id+1}/{self.params.training_epochs} ------")
            self.loss_list = []
            self.model.train()

            for lm_input in loader:
                self.observe_batch(task_id, epoch_id, lm_input)

            if self.loss_list:
                mean_loss = np.mean(self.loss_list)
                if self.accelerator.is_main_process:
                    logger.info(f"[Train] Epoch {epoch_id+1}: mean_loss={mean_loss:.3f}")
                self.accelerator.log({"train_loss": mean_loss}, step=self.global_step)

            # optional dev evaluation
            if (self.params.evaluate_interval > 0) and (epoch_id % self.params.evaluate_interval == 0):
                acc, _ = self.evaluate_current_task(task_id, task_id, "dev", self.params.il_mode)
                if self.accelerator.is_main_process:
                    logger.info(f"[Train] Dev_acc={acc:.3f}")
                self.accelerator.log({f"Dev_Acc_Task_{task_id}": acc}, step=self.global_step)

            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    #                          observe_batch
    # ------------------------------------------------------------------
    def observe_batch(self, task_id, epoch_id, lm_input):
        self.step        += 1
        self.global_step += 1

        student = self.model.module if hasattr(self.model, "module") else self.model
        teacher = (
            self.teacher_model.module if (self.teacher_model and hasattr(self.teacher_model, "module")) else self.teacher_model
        )

        # aliases
        ids_ans, mask_ans, lbls_ans = lm_input["input_ids_with_ans"], lm_input["attention_mask_with_ans"], lm_input["labels_with_ans"]
        ids_gen, mask_gen, lbls_gen = lm_input["input_ids_with_gen_ans"], lm_input["attention_mask_with_gen_ans"], lm_input["labels_with_gen_ans"]
        idx_cil = lm_input["label_idx_cil"]

        # --------------------------------------------------------------
        # Backbone + LM-Head 동시 업데이트 (LAMOL_KD와 동일)
        # --------------------------------------------------------------
        if self.backbone_update:
            # student의 주 device 찾기 (multi-GPU 환경 고려)
            student_device = next(student.parameters()).device
            
            # Input validation to prevent CUDA errors
            vocab_size = student.config.vocab_size
            def validate_input_ids(input_ids, name="input_ids"):
                if input_ids.max() >= vocab_size:
                    logger.warning(f"{name} contains token IDs >= vocab_size ({vocab_size}). Clamping to valid range.")
                    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                if input_ids.min() < 0:
                    logger.warning(f"{name} contains negative token IDs. Clamping to valid range.")
                    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                return input_ids
            
            # Validate all input_ids
            ids_ans = validate_input_ids(ids_ans, "ids_ans")
            ids_gen = validate_input_ids(ids_gen, "ids_gen")
            
            if task_id == 0:
                qa  = student(input_ids=ids_ans, attention_mask=mask_ans, labels=lbls_ans).loss
                gen = student(input_ids=ids_gen, attention_mask=mask_gen, labels=lbls_gen).loss
                loss = qa + self.params.L2KD_lambda * gen
            else:
                cur_mask = idx_cil != -1
                buf_mask = idx_cil == -1

                # 모든 loss tensor를 동일한 device로 초기화
                loss_cur = torch.tensor(0.0, device=student_device)
                if cur_mask.any():
                    cur_ids_ans = validate_input_ids(ids_ans[cur_mask], "cur_ids_ans")
                    cur_ids_gen = validate_input_ids(ids_gen[cur_mask], "cur_ids_gen")
                    qa_c  = student(input_ids=cur_ids_ans, attention_mask=mask_ans[cur_mask], labels=lbls_ans[cur_mask]).loss
                    gen_c = student(input_ids=cur_ids_gen, attention_mask=mask_gen[cur_mask], labels=lbls_gen[cur_mask]).loss
                    loss_cur = (qa_c + self.params.L2KD_lambda * gen_c).to(student_device, non_blocking=True)

                loss_buf = torch.tensor(0.0, device=student_device)
                if buf_mask.any() and teacher is not None:
                    try:
                        # CPU에서 teacher inference (메모리 극대 절약)
                        buf_ids_ans = validate_input_ids(ids_ans[buf_mask], "buf_ids_ans")
                        buf_ids_gen = validate_input_ids(ids_gen[buf_mask], "buf_ids_gen")
                        cpu_ids_ans = buf_ids_ans.cpu()
                        cpu_mask_ans = mask_ans[buf_mask].cpu()
                        cpu_lbls_ans = lbls_ans[buf_mask].cpu()
                        cpu_ids_gen = buf_ids_gen.cpu()
                        cpu_mask_gen = mask_gen[buf_mask].cpu()
                        cpu_lbls_gen = lbls_gen[buf_mask].cpu()
                        
                        with torch.no_grad():
                            t_qa  = teacher(input_ids=cpu_ids_ans, attention_mask=cpu_mask_ans, labels=cpu_lbls_ans).logits
                            t_gen = teacher(input_ids=cpu_ids_gen, attention_mask=cpu_mask_gen, labels=cpu_lbls_gen).logits

                        # 마스킹 후 GPU로 이동
                        keep_qa  = cpu_lbls_ans != -100
                        keep_gen = cpu_lbls_gen != -100
                        
                        if keep_qa.any() and keep_gen.any():
                            t_qa = t_qa[keep_qa].to(student_device, non_blocking=True)
                            t_gen = t_gen[keep_gen].to(student_device, non_blocking=True)

                            # student inference (GPU)
                            with torch.cuda.amp.autocast(enabled=self.accelerator.mixed_precision in ("fp16", "bf16")):
                                s_qa_full = student(input_ids=buf_ids_ans, attention_mask=mask_ans[buf_mask], labels=lbls_ans[buf_mask]).logits
                                s_gen_full= student(input_ids=buf_ids_gen, attention_mask=mask_gen[buf_mask], labels=lbls_gen[buf_mask]).logits
                            
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
                            
                            # KL loss 계산
                            self.kl_loss = self.kl_loss.to(student_device)
                            T = self.params.L2KD_temperature
                            kd_qa  = self.kl_loss(F.log_softmax(s_qa / T, dim=-1), F.softmax(t_qa / T, dim=-1)) * T ** 2
                            kd_gen = self.kl_loss(F.log_softmax(s_gen / T, dim=-1), F.softmax(t_gen / T, dim=-1)) * T ** 2
                            loss_buf = (kd_qa + self.params.L2KD_lambda * kd_gen).to(student_device, non_blocking=True)
                            
                            # 모든 중간 텐서들 정리
                            del t_qa, t_gen, s_qa, s_gen, kd_qa, kd_gen, keep_qa_gpu, keep_gen_gpu, s_qa_full, s_gen_full
                        
                        # CPU 텐서들 정리
                        del cpu_ids_ans, cpu_mask_ans, cpu_lbls_ans, cpu_ids_gen, cpu_mask_gen, cpu_lbls_gen
                        del keep_qa, keep_gen
                        
                    except Exception as e:
                        # Teacher inference 실패 시 warning만 출력하고 계속 진행
                        if self.accelerator.is_main_process:
                            logger.warning(f"[KD] Teacher inference failed: {str(e)}. Proceeding without KD loss.")
                        loss_buf = torch.tensor(0.0, device=student_device)

                # 두 loss 모두 student_device에 있는지 확인
                loss_cur = loss_cur.to(student_device, non_blocking=True)
                loss_buf = loss_buf.to(student_device, non_blocking=True)
                loss = loss_cur + self.params.L2KD_distill_weight * loss_buf

            # backbone + LM-Head 통합 optimizer step
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                self.optimizer.step()
                self.loss_list.append(loss.item())

        # --------------------------------------------------------------
        # logging per steps
        # --------------------------------------------------------------
        if self.params.info_per_steps and self.step % self.params.info_per_steps == 0 and self.loss_list:
            mean_loss = np.mean(self.loss_list)
            if self.accelerator.is_main_process:
                logger.info(f"[Train] Epoch {epoch_id+1}, Step {self.step}: mean_loss={mean_loss:.3f}")
            self.accelerator.log({"train_loss": mean_loss}, step=self.global_step)

    # ======================= Evaluation ============================
    def evaluate_current_task(self, eval_task_id, cur_task_id, phase, il_mode):
        assert phase in ["train", "dev", "test"]
        loaders = (
            self.train_loader_list if phase == "train" else self.dev_loader_list if phase == "dev" else self.test_loader_list
        )
        next_loader = loaders[eval_task_id + 1] if eval_task_id + 1 < len(loaders) else None
        model = self.model.module if hasattr(self.model, "module") else self.model

        # ───── 평가용 LM-Head mini-finetune ─────
        if getattr(self.params, "lm_head_finetune", False):
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
            
            # CRITICAL FIX: model.eval() -> model.train() for fine-tuning
            model.train()
            
            # 디버깅을 위한 로그 추가
            if self.accelerator.is_main_process:
                logger.info(f"[LM-Head FT] Starting fine-tuning for task {eval_task_id}, epochs={self.params.lm_head_epochs}")
            
            for epoch in range(self.params.lm_head_epochs):
                epoch_losses = []
                for lm_input in ft_loader:
                    # Feature extraction with gradient computation disabled for backbone
                    with torch.no_grad(), autocast_ctx():
                        feats = obtain_features(self.params, model, lm_input, self.tokenizer)
                    
                    dev = lm_head.weight.device
                    if feats.device != dev:
                        feats = feats.to(dev, non_blocking=True)

                    # Improved label extraction with better Llama tokenizer handling
                    if "target" in lm_input:
                        # For datasets with explicit target field
                        targets = lm_input["target"]
                        label_tok = []
                        for target in targets:
                            # Tokenize each target individually to ensure proper handling
                            enc = self.tokenizer(target.strip(), add_special_tokens=False, padding=False)
                            if enc["input_ids"]:
                                label_tok.append(enc["input_ids"][0])
                            else:
                                # Fallback if tokenization fails
                                label_tok.append(self.tokenizer.eos_token_id)
                        label_tok = torch.tensor(label_tok, device=dev)
                    else:
                        # Extract from labels_with_ans - find the FIRST non-(-100) token
                        first = []
                        for row in lm_input["labels_with_ans"]:
                            nz = (row != -100).nonzero(as_tuple=False)
                            if nz.numel() > 0:
                                # Get the first valid label token
                                token_id = row[nz[0]].item()
                                # Validate token ID is within vocabulary range
                                if 0 <= token_id < len(self.tokenizer):
                                    first.append(token_id)
                                else:
                                    logger.warning(f"Invalid token ID {token_id} found, using eos_token")
                                    first.append(self.tokenizer.eos_token_id)
                            else:
                                # Fallback to eos token if no valid label found
                                first.append(self.tokenizer.eos_token_id)
                        label_tok = torch.tensor(first, device=dev)

                    # Ensure features require gradient for LM head
                    feats = feats.detach().requires_grad_(True)
                    
                    # Forward pass through LM head
                    logits = lm_head(feats)
                    
                    # Compute loss
                    loss = F.cross_entropy(logits, label_tok)
                    
                    # Check for invalid loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"[LM-Head FT] Invalid loss detected: {loss.item()}")
                        continue
                    
                    epoch_losses.append(loss.item())
                    
                    # Backward pass
                    ft_opt.zero_grad()
                    loss.backward()
                    ft_opt.step()

                    del feats, logits, label_tok, loss
                    torch.cuda.empty_cache()
                
                # Log epoch statistics
                if epoch_losses and self.accelerator.is_main_process:
                    avg_loss = np.mean(epoch_losses)
                    logger.info(f"[LM-Head FT] Epoch {epoch+1}/{self.params.lm_head_epochs}, avg_loss={avg_loss:.4f}")

            # Set model back to eval mode for evaluation
            model.eval()
            
            # Evaluate with fine-tuned LM head
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

            # Restore original LM head state
            lm_head.load_state_dict(orig_state)
            for p in model.parameters():
                p.requires_grad = True
            torch.cuda.empty_cache()
            
            if self.accelerator.is_main_process:
                logger.info(f"[LM-Head FT] Evaluation complete. Acc={acc_cur:.3f}")
            
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

    # ===========================================================================================
    # ======================== Other Model‑Specific Functions ===================================
    def generate_pseudo_buffer_samples(self, task_id: int, num_samples: int) -> List[Dataset]:
        """Generate pseudo samples via generative model (same as original L2KD)."""
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

                cnt_num = num_samples // task_id
                gen_token = f"__{t_id}__" if self.params.LAMOL_use_task_specific_gen_token else "__gen__"

                while cnt_num > 0:
                    gen_n = min(generate_batch_size, cnt_num)
                    lm_in = self.tokenizer([gen_token] * gen_n, padding="longest", return_tensors="pt")
                    lm_in = {k: v.to(model.device) for k, v in lm_in.items()}
                    max_in_len = max(len(ids) for ids in lm_in["input_ids"])
                    gen_ids_all = model.generate(
                        **lm_in,
                        max_new_tokens=self.params.max_seq_length - max_in_len,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=self.params.L2KD_topk,
                    )
                    gen_ids = gen_ids_all[:, max_in_len:]
                    gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=False)

                    for text in gen_texts:
                        if text.count(ans_token) != 1:
                            continue
                        q, a = text.split(ans_token)
                        a = a.replace(self.tokenizer.eos_token, "")
                        sample_dict["input"].append(q)
                        sample_dict["target"].append(a)
                        sample_dict["label_idx_cil"].append(-1)
                        sample_dict["label_idx_til"].append(-1)
                        if self.params.il_mode == "IIL":
                            sample_dict["instance_id"].append(-1)
                            sample_dict["concept_id"].append(-1)
                            sample_dict["relation_id"].append(-1)
                    cnt_num -= gen_n

                if not sample_dict["input"]:
                    logger.info(f"No pseudo samples generated for task {t_id + 1}!")
                    continue

                pseudo_ds = Dataset.from_dict(sample_dict)
                pseudo_ds = pseudo_ds.map(
                    preprocess_function_train_generative_LAMOL,
                    batched=True,
                    desc=f"Generate pseudo samples for task {t_id + 1}",
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
                if self.params.il_mode == "IIL":
                    pseudo_ds.set_format(
                        type="torch",
                        columns=[
                            "input_ids",
                            "attention_mask",
                            "label_idx_cil",
                            "label_idx_til",
                            "input_ids_with_ans",
                            "attention_mask_with_ans",
                            "labels_with_ans",
                            "input_ids_with_gen_ans",
                            "attention_mask_with_gen_ans",
                            "labels_with_gen_ans",
                            "target",
                            "instance_id",
                            "concept_id",
                            "relation_id",
                        ],
                    )
                else:
                    pseudo_ds.set_format(
                        type="torch",
                        columns=[
                            "input_ids",
                            "attention_mask",
                            "label_idx_cil",
                            "label_idx_til",
                            "input_ids_with_ans",
                            "attention_mask_with_ans",
                            "labels_with_ans",
                            "input_ids_with_gen_ans",
                            "attention_mask_with_gen_ans",
                            "labels_with_gen_ans",
                        ],
                    )

                pseudo_dataset_list.append(pseudo_ds)

        return pseudo_dataset_list

    # ===========================================================================================