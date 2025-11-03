import numpy as np
import logging
import torch
import torch.nn as nn
from typing import List
from copy import deepcopy
from datasets import Dataset
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F
from transformers import Conv1D
from transformers.activations import gelu
from contextlib import nullcontext

from utils.metric import ResultSummary, ResultSummary2
from utils.backbone import get_backbone, obtain_features
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader, preprocess_function_train_and_predict_generative_PCLL
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_sent_level_acc_with_generation, evaluate_sent_level_acc_with_lmhead
from utils.prompt import get_prompt_PCLL
from models.Base import BaseLearner
from torch.optim import AdamW

logger = logging.getLogger()

def get_PCLL_params(parser):
    '''
        The parameters of model PCLL
    '''

    parser.add_argument("--PCLL_KD_lambda", type=float, default=0.50, help="The weight of the generation target in PCLL")
    parser.add_argument("--PCLL_weight_vae_loss", type=float, default=0, help="The weight of vae_loss (the model can be normally trained only when it is zero!?)")
    parser.add_argument("--PCLL_weight_gen_loss", type=float, default=0.50, help="The weight of generation_loss")
    parser.add_argument("--PCLL_alpha_z", type=float, default=0.1, help="Multiply alpha when adding the latent z embedding onto the original embeddings.")
    parser.add_argument("--PCLL_KD_gamma", type=float, default=0.20, help="The ratio of psesudo old samples w.r.t the training data of new task.")
    parser.add_argument("--PCLL_KD_topk", type=int, default=20, help="The top-k sampling for generating psesudo old samples.")
    parser.add_argument("--PCLL_KD_temperature", type=float, default=1.0, help="The temperature for knowledge distillation.")
    
    # =============== LM-Head 관련 파라미터 ===============
    parser.add_argument("--pooling_type", type=str, default='mean', choices=['cls', 'mean'], help="Pooling type for feature extraction")

class PCLL(BaseLearner):
    '''
        PCLL: PCLL is build upon LAMOL. Furthermore, PCLL utilizes the targets of varational autoencoders 
        and word-level knowledge distillation to train generative models. 
        The implementation is based on the official released code in the [repository](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/pcll).

        - [Prompt Conditioned VAE: Enhancing Generative Replay for Lifelong Learning in Task-Oriented Dialogue](https://aclanthology.org/2022.emnlp-main.766/)
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)

        assert params.classifier in ['None'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'PCLL')
        assert params.il_mode in ['IIL','CIL','TIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'PCLL')
        assert params.classification_type == 'sentence-level', 'NotImplemented for classification type %s'%(params.classification_type)
        assert params.backbone_type == 'generative', 'NotImplemented for backbone type %s'%(params.backbone_type)
        assert not params.is_replay, 'NotImplemented for is_replay = %s'%(params.is_replay)

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.probing_result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.kmeans_result_summary = ResultSummary2(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.gmm_result_summary = ResultSummary2(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.spectral_result_summary = ResultSummary2(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.deep_clustering_result_summary = ResultSummary2(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.agglomerative_result_summary = ResultSummary2(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        if self.params.il_mode == 'IIL':
            self.result_summary_train = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        backbone_model, self.tokenizer = get_backbone(self.params)
        self.model = PCLLModel(backbone_model, self.params)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def build_classifier(self):
        self.classifier_list = None

    def build_optimizer(self):
        """
        • Backbone + (optional) classifier용 옵티마이저  
        """
        # ── 1) 백본‧분류기 옵티마이저 ─────────────────
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None
    
    def accelerate_prepare(self):
        self.wrap_model = WrapModel(self.model, self.classifier_list)
        self.wrap_model, self.optimizer, *self.train_loader_list = self.accelerator.prepare(self.wrap_model, self.optimizer, *self.train_loader_list)
        self.dev_loader_list = self.accelerator.prepare(*self.dev_loader_list)
        self.test_loader_list = self.accelerator.prepare(*self.test_loader_list)
        self.wrap_teacher_model = None
    # =============================================================================================

    # ================================= Task-Level Functions =======================================
    def begin_task(self, task_id):
        super().begin_task(task_id)
        
        # Prepare teacher model (메모리 효율적 방식)
        if task_id > 0:
            # ── 이전 teacher 정리 ──────────────────────────────────────────
            if getattr(self, "wrap_teacher_model", None) is not None:
                del self.wrap_teacher_model
                self.wrap_teacher_model = None
                torch.cuda.empty_cache()

            # ── teacher 모델을 메모리 효율적으로 생성 ──
            from utils.backbone import get_backbone
            import gc
            
            # 1) 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()
            
            # 2) teacher backbone을 새로 로드 (CPU에서)
            teacher_backbone, _ = get_backbone(self.params)
            teacher_model = PCLLModel(teacher_backbone, self.params)
            
            # 3) teacher model 전체를 CPU로 이동
            teacher_model = teacher_model.cpu()
            
            # 4) student의 state_dict를 teacher에 복사 (메모리 효율적)
            student_model = self.wrap_model.module.model if hasattr(self.wrap_model, "module") else self.wrap_model.model
            
            # state_dict 복사 시 메모리 효율화
            with torch.no_grad():
                for (s_name, s_param), (t_name, t_param) in zip(
                    student_model.named_parameters(), 
                    teacher_model.named_parameters()
                ):
                    if s_name == t_name:  # 파라미터 이름 매칭
                        t_param.copy_(s_param.cpu())
            
            # use_cache=False로 설정하여 gradient checkpointing과 호환
            if hasattr(teacher_model.backbone_model, 'config'):
                teacher_model.backbone_model.config.use_cache = False
            
            # 5) teacher model의 모든 컴포넌트를 명시적으로 CPU로 이동
            teacher_model.backbone_model = teacher_model.backbone_model.cpu()
            teacher_model.prior_mean = teacher_model.prior_mean.cpu()
            teacher_model.prior_logvar = teacher_model.prior_logvar.cpu()
            teacher_model.post_mean = teacher_model.post_mean.cpu()
            teacher_model.post_logvar = teacher_model.post_logvar.cpu()
            teacher_model.avg_attn = teacher_model.avg_attn.cpu()
            teacher_model.latent_mlp = teacher_model.latent_mlp.cpu()
            teacher_model.kl_loss = teacher_model.kl_loss.cpu() if hasattr(teacher_model.kl_loss, 'cpu') else teacher_model.kl_loss
            teacher_model.ce_loss = teacher_model.ce_loss.cpu() if hasattr(teacher_model.ce_loss, 'cpu') else teacher_model.ce_loss
            
            # teacher는 CPU에 유지 (메모리 절약)
            self.wrap_teacher_model = teacher_model
            
            # 메모리 정리
            del teacher_backbone
            torch.cuda.empty_cache()
            gc.collect()

    def end_task(self, task_id):
        super().end_task(task_id)

    # ==============================================================================================

    # ================================= Epoch-Level Functions =======================================
    def train_epochs(self, task_id):
        '''
            Training the model with serveral epochs
        '''

        if task_id>0:
            train_dataset = self.train_loader_list[task_id].dataset
            pseudo_buf_dataset_list = self.generate_pseudo_buffer_samples(task_id=task_id,
                                                                     num_samples=int(len(train_dataset)*self.params.PCLL_KD_gamma))
            pseudo_train_loader = DataLoader(
                ConcatDataset(pseudo_buf_dataset_list),
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False
            )
            pseudo_train_loader = self.accelerator.prepare(pseudo_train_loader)
            pseudo_train_loader_iter = iter(pseudo_train_loader)
            num_batch_pseudo = len(pseudo_train_loader)
            pseudo_idx = 0
        
        cur_train_loader = self.train_loader_list[task_id]

        total_epochs = self.params.training_epochs

        self.beta_list = self.wrap_model.model.frange_cycle_linear(int(total_epochs*len(cur_train_loader)))

        for epoch_id in range(total_epochs):

            if self.accelerator.is_main_process:
                logger.info("------------------------ epoch %d ------------------------" %(epoch_id+1))

            self.begin_epoch(task_id, epoch_id)

            for lm_input in cur_train_loader:
                if task_id==0:
                    self.observe_batch(task_id, epoch_id, lm_input, buf_lm_input=None) 
                else:
                    if pseudo_idx==num_batch_pseudo:
                        pseudo_train_loader_iter = iter(pseudo_train_loader)
                        pseudo_idx = 0

                    buf_lm_input = next(pseudo_train_loader_iter) 
                    pseudo_idx += 1

                    self.observe_batch(task_id, epoch_id, lm_input, buf_lm_input=buf_lm_input) 

            self.end_epoch(task_id, epoch_id)
            
            # 에포크 종료 후 메모리 정리
            torch.cuda.empty_cache()

    def begin_epoch(self, task_id, epoch_id):
        '''
            Start of each epoch
        '''
        # Avoid overwrite the result of the same global step
        if ((self.params.evaluate_interval>0) and (epoch_id>0 and epoch_id%self.params.evaluate_interval==0)) or \
            (task_id==0 and epoch_id==0):
            self.evaluate_model(task_id=task_id)
        self.loss_list = []
        self.wrap_model.train()

    def observe_batch(self, task_id, epoch_id, lm_input, buf_lm_input=None):
        '''
            Observe a batch of data (메모리 최적화 버전)
        '''
        # Update step
        self.step += 1
        self.global_step += 1

        # For Distributed Data Parallel
        if hasattr(self.wrap_model,'module'):
            model = self.wrap_model.module.model
        else:
            model = self.wrap_model.model
        
        # Student device 확인
        student_device = next(model.parameters()).device
        
        # Mixed precision context
        autocast_ctx = (
            torch.cuda.amp.autocast
            if self.accelerator.mixed_precision in ("fp16", "bf16")
            else nullcontext
        )

        # Beta for VAE
        beta = self.beta_list[self.step-1]

        # ====================== Current Task Loss ======================
        with autocast_ctx():
            vae_loss = model.compute_vae_loss(lm_input=lm_input, beta=beta)
            lm_loss = model.compute_lm_loss(lm_input=lm_input)
            total_loss = self.params.PCLL_weight_vae_loss * vae_loss + lm_loss

        # 중간 텐서 정리
        del vae_loss, lm_loss

        # ====================== Knowledge Distillation (메모리 효율적) ======================
        if task_id > 0 and buf_lm_input is not None and self.wrap_teacher_model is not None:
            teacher_model = self.wrap_teacher_model
            
            try:
                # Teacher inference를 완전히 CPU에서 수행
                teacher_loss = self._safe_teacher_inference(teacher_model, buf_lm_input, beta)
                
                # Student inference on GPU
                with autocast_ctx():
                    student_distil_vae_loss = model.compute_vae_loss(
                        lm_input=buf_lm_input, 
                        teacher_backbone_model=None, 
                        beta=beta
                    )
                    student_distil_lm_loss = model.compute_lm_loss(
                        lm_input=buf_lm_input, 
                        teacher_backbone_model=None
                    )
                
                student_total_distil = self.params.PCLL_weight_vae_loss * student_distil_vae_loss + student_distil_lm_loss
                
                # Teacher loss가 유효한 경우만 distillation 적용
                if teacher_loss is not None:
                    teacher_loss_gpu = teacher_loss.to(student_device, non_blocking=True)
                    distil_loss = F.mse_loss(student_total_distil, teacher_loss_gpu.detach())
                    total_loss = total_loss + distil_loss
                    del teacher_loss_gpu, distil_loss
                else:
                    # Teacher inference 실패 시 regularization loss만 적용
                    distil_loss = 0.1 * student_total_distil
                    total_loss = total_loss + distil_loss
                    del distil_loss
                
                # 중간 텐서들 정리
                del student_distil_vae_loss, student_distil_lm_loss, student_total_distil
                if teacher_loss is not None:
                    del teacher_loss
                
            except Exception as e:
                if self.accelerator.is_main_process:
                    logger.warning(f"Distillation failed: {e}, using only current task loss")
            finally:
                torch.cuda.empty_cache()

        # ====================== Backbone 업데이트 ======================
        model.train()
        self.optimizer.zero_grad()        
        self.accelerator.backward(total_loss)
        
        scalar_loss = total_loss.item()
        if not(np.isnan(scalar_loss)) or not(np.isinf(scalar_loss)):
            self.optimizer.step()
            self.loss_list.append(scalar_loss)

        # 최종 메모리 정리
        del total_loss
        torch.cuda.empty_cache()

        # Print training information
        if self.params.info_per_steps and self.step%self.params.info_per_steps==0:
            if self.accelerator.is_main_process:
                mean_loss = np.mean(self.loss_list) if self.loss_list else 0.0
                logger.info(f"Epoch {epoch_id+1}, Step {self.step}: mean_loss = {mean_loss:.3f}")
                self.accelerator.log({"train_loss": mean_loss}, step=self.global_step)

    def _safe_teacher_inference(self, teacher_model, buf_lm_input, beta):
        """
        Teacher model을 완전히 CPU에서 안전하게 실행
        """
        try:
            # Teacher model이 CPU에 있는지 확인
            teacher_device = next(teacher_model.parameters()).device
            if teacher_device.type != 'cpu':
                teacher_model = teacher_model.cpu()
                torch.cuda.empty_cache()
            
            # 모든 입력 텐서를 CPU로 변환 (깊은 복사)
            cpu_buf_input = {}
            for k, v in buf_lm_input.items():
                if isinstance(v, torch.Tensor):
                    # 완전히 분리된 CPU 텐서 생성
                    cpu_buf_input[k] = v.detach().cpu().clone()
                else:
                    cpu_buf_input[k] = v
            
            # Teacher model의 모든 컴포넌트를 CPU로 강제 이동
            self._ensure_teacher_on_cpu(teacher_model)
            
            # CPU에서 teacher inference 수행
            with torch.no_grad():
                teacher_model.eval()
                
                # VAE loss 계산 (CPU)
                teacher_vae_loss = self._compute_vae_loss_cpu(teacher_model, cpu_buf_input, beta)
                
                # LM loss 계산 (CPU)
                teacher_lm_loss = self._compute_lm_loss_cpu(teacher_model, cpu_buf_input)
                
                # 총 teacher loss 계산
                teacher_total_loss = self.params.PCLL_weight_vae_loss * teacher_vae_loss + teacher_lm_loss
                
                # 결과를 CPU에서 분리된 텐서로 반환
                result = teacher_total_loss.detach().clone()
                
            # 메모리 정리
            del cpu_buf_input, teacher_vae_loss, teacher_lm_loss, teacher_total_loss
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            if self.accelerator.is_main_process:
                logger.warning(f"Safe teacher inference failed: {e}")
            return None
    
    def _ensure_teacher_on_cpu(self, teacher_model):
        """Teacher model의 모든 컴포넌트를 CPU로 강제 이동"""
        components = [
            'backbone_model', 'prior_mean', 'prior_logvar', 
            'post_mean', 'post_logvar', 'avg_attn', 'latent_mlp'
        ]
        
        for comp_name in components:
            if hasattr(teacher_model, comp_name):
                comp = getattr(teacher_model, comp_name)
                if comp is not None:
                    try:
                        setattr(teacher_model, comp_name, comp.cpu())
                    except:
                        pass  # 이미 CPU에 있거나 이동할 수 없는 경우
    
    def _compute_vae_loss_cpu(self, teacher_model, cpu_lm_input, beta):
        """CPU에서 VAE loss 계산 (단순화 버전)"""
        # 기본적인 reconstruction loss만 계산 (KL divergence 생략하여 안전성 확보)
        try:
            loss = teacher_model.backbone_model(
                input_ids=cpu_lm_input['input_ids_prompt_input'],
                attention_mask=cpu_lm_input['attention_mask_prompt_input'],
                labels=cpu_lm_input['labels_gen_prompt_input']
            ).loss
            return loss
        except:
            # 실패 시 더미 loss 반환
            return torch.tensor(0.0, dtype=torch.float32)
    
    def _compute_lm_loss_cpu(self, teacher_model, cpu_lm_input):
        """CPU에서 LM loss 계산 (단순화 버전)"""
        try:
            qa_loss = teacher_model.backbone_model(
                input_ids=cpu_lm_input['input_ids_prompt_input_anstoken_ans'],
                attention_mask=cpu_lm_input['attention_mask_prompt_input_anstoken_ans'],
                labels=cpu_lm_input['labels_qa_prompt_input_anstoken_ans']
            ).loss
            return qa_loss
        except:
            # 실패 시 더미 loss 반환
            return torch.tensor(0.0, dtype=torch.float32)

    def end_epoch(self, task_id, epoch_id):
        '''
            End of each epoch
        '''
        # Print training information
        if len(self.loss_list)>0:
            mean_loss = np.mean(self.loss_list)
            if self.accelerator.is_main_process:
                logger.info("Epoch %d, Step %d: Total_loss=%.3f"%(
                            epoch_id+1, self.step, mean_loss
                    ))
            self.accelerator.log({'loss':mean_loss},step=self.global_step)
            
        # For evaluation
        if (self.params.evaluate_interval>0) and epoch_id%self.params.evaluate_interval==0:
            il_mode = self.params.il_mode
            if il_mode == 'IIL':
                acc, save_dict = self.evaluate_current_task(task_id, task_id, 'dev', il_mode)
            else:
                acc = self.evaluate_current_task(task_id, task_id, 'dev', il_mode)
            if self.accelerator.is_main_process:
                logger.info("Mode %s, Current Task %d, Epoch %d, Step %d: Dev_acc=%.3f" % (
                    il_mode, task_id, epoch_id+1, self.step, acc
                ))
            self.accelerator.log({'Dev_Acc_Task_%d'%(task_id):acc},step=self.global_step)
            dev_score = acc

            if dev_score > self.best_score:
                if self.accelerator.is_main_process:
                    logger.info("Find better model!!")

        # 더 적극적인 GPU 메모리 정리
        if hasattr(torch.cuda, 'synchronize'):
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Python 가비지 컬렉터 실행
        import gc
        gc.collect()
    # ===========================================================================================


    # ================== Evaluation, Logging, Saving and Loading Functions ======================
    def evaluate_current_task(self, eval_task_id, cur_task_id, phase, il_mode):
        """
        L2KD 스타일의 평가 방식을 따라 구현
        """
        assert phase in ["train", "dev", "test"]
        loaders = (
            self.train_loader_list if phase == "train" 
            else self.dev_loader_list if phase == "dev" 
            else self.test_loader_list
        )
        
        # ── DDP unwrap ───────────────────────────────
        wrap_model = (
            self.wrap_model.module if hasattr(self.wrap_model, "module") else self.wrap_model
        )
        model = wrap_model.model.backbone_model  # PCLL은 PCLLModel 안에 backbone_model이 있음

        # ───── 평가용 LM-Head 미세조정 후 평가 ─────
        if getattr(self.params, "lm_head_finetune", False):
            if self.accelerator.is_main_process:
                logger.info(f"PCLL evaluation: using lm_head_finetune method")
                
            lm_head = model.get_output_embeddings()
            orig_state = deepcopy(lm_head.state_dict())  # 원본 상태 저장

            # backbone freeze, lm_head만 trainable
            for p in model.parameters():
                p.requires_grad = False
            for p in lm_head.parameters():
                p.requires_grad = True

            # 현재 task 데이터로 lm_head 미세조정
            ft_loader = self.train_loader_list[eval_task_id]
            ft_opt = torch.optim.AdamW(lm_head.parameters(), lr=getattr(self.params, "lm_head_lr", 1e-4))

            autocast_ctx = (
                torch.cuda.amp.autocast
                if self.accelerator.mixed_precision in ("fp16", "bf16")
                else nullcontext
            )
            
            model.eval()
            for epoch in range(getattr(self.params, "lm_head_epochs", 1)):
                if self.accelerator.is_main_process:
                    logger.info(f"LM head fine-tuning epoch {epoch + 1}")
                    
                for batch_idx, lm_input in enumerate(ft_loader):
                    try:
                        # PCLL prompt 형식으로 input 변환
                        if 'input' in lm_input:
                            input_texts = lm_input['input']
                        else:
                            input_texts = self.tokenizer.batch_decode(lm_input['input_ids'], skip_special_tokens=True)
                        
                        # PCLL prompt 생성 (훈련 시와 동일한 형식)
                        prompted_inputs = []
                        for text in input_texts:
                            clean_text = text.strip()
                            # PCLL 훈련 시 사용하는 prompt 형식과 동일하게
                            prompt_template = f'In the current task, which intent category best describes: " {clean_text} "? Answer:'
                            prompted_inputs.append(prompt_template)
                        
                        # Prompted input tokenize
                        prompted_lm_input = self.tokenizer(
                            prompted_inputs,
                            padding='longest',
                            return_tensors='pt',
                            truncation=True,
                            max_length=512
                        )
                        prompted_lm_input = {k: v.to(model.device) for k, v in prompted_lm_input.items()}
                        
                        with torch.no_grad(), autocast_ctx():
                            feats = obtain_features(self.params, model, prompted_lm_input, self.tokenizer)
                        
                        dev = lm_head.weight.device
                        if feats.device != dev:
                            feats = feats.to(dev, non_blocking=True)

                        # PCLL 전용: labels_qa_prompt_input_anstoken_ans에서 첫 번째 유효한 토큰 추출
                        if "labels_qa_prompt_input_anstoken_ans" in lm_input:
                            first = []
                            for row in lm_input["labels_qa_prompt_input_anstoken_ans"]:
                                nz = (row != -100).nonzero(as_tuple=False)
                                first.append(row[nz[0]].item() if nz.numel() else self.tokenizer.eos_token_id)
                            label_tok = torch.tensor(first, device=dev)
                        elif "target" in lm_input:
                            enc = self.tokenizer(lm_input["target"], add_special_tokens=False, padding=False)
                            label_tok = torch.tensor([ids[0] for ids in enc["input_ids"]], device=dev)
                        else:
                            # label_idx_cil을 이용해서 idx2label로 변환
                            label_idx = lm_input['label_idx_cil'].cpu().numpy()
                            idx2label = self.CL_dataset.continual_config["idx2label"]
                            target_labels = [idx2label[int(l)] for l in label_idx if int(l) < len(idx2label)]
                            enc = self.tokenizer(target_labels, add_special_tokens=False, padding=False)
                            label_tok = torch.tensor([ids[0] for ids in enc["input_ids"]], device=dev)

                        logits = lm_head(feats)
                        loss = F.cross_entropy(logits, label_tok)
                        ft_opt.zero_grad()
                        loss.backward()
                        ft_opt.step()

                        if batch_idx == 0 and epoch == 0 and self.accelerator.is_main_process:
                            logger.info(f"LM head fine-tuning - first batch loss: {loss.item():.4f}")
                            logger.info(f"Sample prompt for lm_head training: '{prompted_inputs[0] if prompted_inputs else 'N/A'}'")
                            logger.info(f"Sample target token: {label_tok[0].item() if len(label_tok) > 0 else 'N/A'}")

                        del feats, logits, label_tok, loss
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        if self.accelerator.is_main_process:
                            logger.error(f"Error in lm_head fine-tuning: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                        continue

            # 미세조정된 lm_head로 평가 (PCLL 전용)
            if self.accelerator.is_main_process:
                logger.info("Evaluating with fine-tuned lm_head...")
                
            acc = self._evaluate_pcll_lmhead(
                model=model,
                eval_data_loader=loaders[eval_task_id],
                tokenizer=self.tokenizer,
                accelerator=self.accelerator,
                params=self.params,
                idx2label=self.CL_dataset.continual_config["idx2label"],
            )

            # 원본 lm_head 상태로 복구 (다음 task 훈련을 위해)
            lm_head.load_state_dict(orig_state)
            for p in model.parameters():
                p.requires_grad = True
            torch.cuda.empty_cache()
            
            if self.accelerator.is_main_process:
                logger.info(f"PCLL lm_head evaluation result: accuracy = {acc}")
            
            if il_mode == 'IIL':
                return acc, {"accuracy": acc}
            else:
                return acc, None
        
        else:
            # ───── 기본 generation 평가 ─────
            if self.accelerator.is_main_process:
                logger.info(f"PCLL evaluation: using generation method")
                logger.info(f"eval_task_id={eval_task_id}, cur_task_id={cur_task_id}, phase={phase}, il_mode={il_mode}")
                logger.info(f"Data loader size: {len(loaders[eval_task_id])}")
                
            acc = self._evaluate_pcll_generation(
                model=model,
                eval_data_loader=loaders[eval_task_id],
                tokenizer=self.tokenizer,
                accelerator=self.accelerator,
                params=self.params,
                idx2label=self.CL_dataset.continual_config["idx2label"],
            )
            
            if self.accelerator.is_main_process:
                logger.info(f"PCLL generation evaluation result: accuracy = {acc}")
        
        if il_mode == 'IIL':
            # IIL mode에서는 dictionary 형태로 반환 (성능 추적을 위해)
            return acc, {"accuracy": acc}
        else:
            return acc, None


    
    def _evaluate_pcll_generation(self, model, eval_data_loader, tokenizer, accelerator, params, idx2label):
        """
        PCLL 전용 generation 평가 함수
        """
        if accelerator.is_main_process:
            logger.info("Starting PCLL generation evaluation...")
            
        model.eval()
        acc_list_all = []
        batch_count = 0
        
        try:
            with torch.no_grad():
                for lm_input in eval_data_loader:
                    batch_count += 1
                    batch_size = lm_input['input_ids'].shape[0]
                    
                    if accelerator.is_main_process and batch_count == 1:
                        logger.info(f"First batch size: {batch_size}")
                        logger.info(f"Available keys in lm_input: {list(lm_input.keys())}")
                    
                    # Input text 추출
                    if 'input' in lm_input:
                        input_texts = lm_input['input']
                    else:
                        # tokenized input을 decode
                        input_texts = tokenizer.batch_decode(lm_input['input_ids'], skip_special_tokens=True)
                    
                    if accelerator.is_main_process and batch_count == 1:
                        logger.info(f"Sample input text: '{input_texts[0] if input_texts else 'NO TEXT'}'")
                    
                    # PCLL 형식의 prompt 생성
                    prompted_inputs = []
                    for text in input_texts:
                        # 기본적인 텍스트 정리
                        clean_text = text.strip()
                        # prompt 생성
                        prompt_template = f'In the current task, which intent category best describes: " {clean_text} "? Answer:'
                        prompted_inputs.append(prompt_template)
                    
                    if accelerator.is_main_process and batch_count == 1:
                        logger.info(f"Sample prompt: '{prompted_inputs[0] if prompted_inputs else 'NO PROMPT'}'")
                    
                    # Tokenize prompted inputs
                    try:
                        prompted_lm_input = tokenizer(
                            prompted_inputs,
                            padding='longest',
                            return_tensors='pt',
                            truncation=True,
                            max_length=512
                        )
                        prompted_lm_input = {k: v.to(model.device) for k, v in prompted_lm_input.items()}
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.error(f"Tokenization error: {e}")
                        continue
                    
                    # Generation
                    try:
                        input_len = prompted_lm_input['input_ids'].shape[1]
                        max_new_tokens = min(20, getattr(params, 'backbone_max_new_token', 20))  # 더 짧게
                        
                        with torch.no_grad():  # 메모리 절약
                            generate_ids_all = model.generate(
                                **prompted_lm_input,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=False,  # greedy decoding
                                temperature=1.0,
                                num_beams=1,
                                use_cache=True
                            )
                        
                        generate_ids = generate_ids_all[:, input_len:].contiguous()
                        
                        # 중간 텐서 정리
                        del prompted_lm_input, generate_ids_all
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.error(f"Generation error: {e}")
                        # 메모리 정리 후 continue
                        if 'prompted_lm_input' in locals():
                            del prompted_lm_input
                        torch.cuda.empty_cache()
                        continue
                    
                    # DDP 처리
                    try:
                        generate_ids = accelerator.pad_across_processes(
                            generate_ids, dim=1,
                            pad_index=tokenizer.eos_token_id,
                            pad_first=False
                        )
                        label_idx = lm_input['label_idx_cil']
                        generate_ids, label_idx = accelerator.gather_for_metrics((generate_ids, label_idx))
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.error(f"DDP gathering error: {e}")
                        continue
                    
                    # Decode predictions
                    try:
                        generate_seq = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
                        generate_seq = [
                            pred.strip(' \n').lower().split(tokenizer.eos_token)[0].strip()
                            for pred in generate_seq
                        ]
                        
                        if accelerator.is_main_process and batch_count == 1:
                            logger.info(f"Sample generated text: '{generate_seq[0] if generate_seq else 'NO GENERATION'}'")
                            
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.error(f"Decoding error: {e}")
                        continue
                    
                    # Compare with targets
                    try:
                        if not idx2label or idx2label == []:
                            # raw string targets
                            if 'target' in lm_input:
                                target_output = lm_input['target']
                                for pred, target in zip(generate_seq, target_output):
                                    match = pred.strip().lower() == target.strip().lower()
                                    acc_list_all.append(1 if match else 0)
                                    
                                    # 디버깅용 로깅 (첫 번째 배치만)
                                    if batch_count == 1 and len(acc_list_all) <= 3 and accelerator.is_main_process:
                                        logger.info(f"PCLL Debug - Pred: '{pred}', Target: '{target}', Match: {match}")
                        else:
                            # class-index targets
                            label_np = label_idx.detach().cpu().numpy()
                            target_output = [idx2label[int(l)] for l in label_np if int(l) < len(idx2label)]
                            target_output = [t.strip(' \n').lower() for t in target_output]
                            
                            if accelerator.is_main_process and batch_count == 1:
                                logger.info(f"Sample target: '{target_output[0] if target_output else 'NO TARGET'}'")
                            
                            for pred, target in zip(generate_seq[:len(target_output)], target_output):
                                match = pred == target
                                acc_list_all.append(1 if match else 0)
                                
                                # 디버깅용 로깅 (첫 번째 배치만)
                                if batch_count == 1 and len(acc_list_all) <= 3 and accelerator.is_main_process:
                                    logger.info(f"PCLL Debug - Pred: '{pred}', Target: '{target}', Match: {match}")
                                    
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.error(f"Target comparison error: {e}")
                        continue
                        
                    # 진행 상황 로깅
                    if batch_count % 10 == 0 and accelerator.is_main_process:
                        current_acc = np.mean(acc_list_all) * 100 if acc_list_all else 0.0
                        logger.info(f"PCLL Evaluation progress: {batch_count} batches, current acc: {current_acc:.2f}%")
                        
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"PCLL evaluation error: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        acc = np.round(np.mean(acc_list_all) * 100, 3) if acc_list_all else 0.0
        
        if accelerator.is_main_process:
            logger.info(f"PCLL evaluation completed: {len(acc_list_all)} samples, accuracy: {acc}%")
            
        model.train()
        return acc
    
    def _evaluate_pcll_lmhead(self, model, eval_data_loader, tokenizer, accelerator, params, idx2label):
        """
        PCLL 전용 lm_head 평가 함수
        """
        if accelerator.is_main_process:
            logger.info("Starting PCLL lm_head evaluation...")
            
        model.eval()
        lm_head = model.get_output_embeddings()
        acc_list_all = []
        batch_count = 0
        
        try:
            with torch.no_grad():
                for lm_input in eval_data_loader:
                    batch_count += 1
                    batch_size = lm_input['input_ids'].shape[0]
                    
                    if accelerator.is_main_process and batch_count == 1:
                        logger.info(f"LM head eval - first batch size: {batch_size}")
                    
                    # Input text 추출
                    if 'input' in lm_input:
                        input_texts = lm_input['input']
                    else:
                        input_texts = tokenizer.batch_decode(lm_input['input_ids'], skip_special_tokens=True)
                    
                    # PCLL prompt 생성 (훈련 시와 동일한 형식)
                    prompted_inputs = []
                    for text in input_texts:
                        clean_text = text.strip()
                        prompt_template = f'In the current task, which intent category best describes: " {clean_text} "? Answer:'
                        prompted_inputs.append(prompt_template)
                    
                    if accelerator.is_main_process and batch_count == 1:
                        logger.info(f"Sample prompt for lm_head eval: '{prompted_inputs[0] if prompted_inputs else 'N/A'}'")
                    
                    # Tokenize prompted inputs
                    try:
                        prompted_lm_input = tokenizer(
                            prompted_inputs,
                            padding='longest',
                            return_tensors='pt',
                            truncation=True,
                            max_length=512
                        )
                        prompted_lm_input = {k: v.to(model.device) for k, v in prompted_lm_input.items()}
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.error(f"Tokenization error in lm_head eval: {e}")
                        continue
                    
                    # Feature extraction
                    try:
                        feats = obtain_features(params, model, prompted_lm_input, tokenizer)
                        
                        # LM head prediction
                        logits = lm_head(feats)
                        preds = logits.argmax(dim=-1)
                        
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.error(f"Feature extraction or prediction error: {e}")
                        continue
                    
                    # Target token 추출
                    try:
                        if "labels_qa_prompt_input_anstoken_ans" in lm_input:
                            # PCLL 전용 target 추출
                            gold_tokens = []
                            for row in lm_input["labels_qa_prompt_input_anstoken_ans"]:
                                nz = (row != -100).nonzero(as_tuple=False)
                                gold_tokens.append(row[nz[0]].item() if nz.numel() else tokenizer.eos_token_id)
                            gold = torch.tensor(gold_tokens, device=preds.device)
                        elif "target" in lm_input:
                            enc = tokenizer(lm_input["target"], add_special_tokens=False, padding=False)
                            gold = torch.tensor([ids[0] for ids in enc["input_ids"]], device=preds.device)
                        else:
                            # label_idx_cil을 이용해서 idx2label로 변환
                            label_idx = lm_input['label_idx_cil'].cpu().numpy()
                            target_labels = [idx2label[int(l)] for l in label_idx if int(l) < len(idx2label)]
                            enc = tokenizer(target_labels, add_special_tokens=False, padding=False)
                            gold = torch.tensor([ids[0] for ids in enc["input_ids"]], device=preds.device)
                        
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.error(f"Target extraction error: {e}")
                        continue
                    
                    # DDP 처리
                    try:
                        preds, gold = accelerator.gather_for_metrics((preds, gold))
                        
                        # Accuracy 계산
                        correct = (preds == gold).sum().item()
                        total = gold.numel()
                        acc_list_all.extend([1] * correct + [0] * (total - correct))
                        
                        # 디버깅용 로깅 (첫 번째 배치만)
                        if batch_count == 1 and accelerator.is_main_process:
                            logger.info(f"LM head eval - Sample pred token: {preds[0].item()}, target token: {gold[0].item()}")
                            logger.info(f"LM head eval - Batch accuracy: {correct}/{total} = {correct/total*100:.2f}%")
                            
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.error(f"DDP or accuracy calculation error: {e}")
                        continue
                        
                    # 진행 상황 로깅
                    if batch_count % 10 == 0 and accelerator.is_main_process:
                        current_acc = np.mean(acc_list_all) * 100 if acc_list_all else 0.0
                        logger.info(f"LM head eval progress: {batch_count} batches, current acc: {current_acc:.2f}%")
                        
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"PCLL lm_head evaluation error: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        acc = np.round(np.mean(acc_list_all) * 100, 3) if acc_list_all else 0.0
        
        if accelerator.is_main_process:
            logger.info(f"PCLL lm_head evaluation completed: {len(acc_list_all)} samples, accuracy: {acc}%")
            
        model.train()
        return acc

    # ===========================================================================================

    # ======================== Other Model-Specific Functions ===================================
    def generate_pseudo_buffer_samples(self, task_id: int, num_samples: int) -> List[Dataset]:
        '''
            Generate pseudo old samples with generative models

            Args:
                - task_id: the current task id
                - num_samples: the number of samples to be generated

            Return:
                pseudo_dataset_list
        '''
        input_column = 'input'
        target_column = 'target'

        gen_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        num_task = self.CL_dataset.continual_config['NUM_TASK']

        pseudo_dataset_list = []
        
        generate_batch_size = 4  # 메모리 절약을 위해 배치 크기 줄임

        # For Distributed Data Parallel
        if hasattr(self.wrap_model,'module'):
            model = self.wrap_model.module.model
        else:
            model = self.wrap_model.model

        with torch.no_grad():
        
            for t_id in range(task_id):

                if self.params.il_mode == 'IIL':
                    pesudo_samples_dict = {
                        'input': [], 'target': [], 'label_idx_cil': [], 'label_idx_til': [],
                        'instance_id': [], 'concept_id': [], 'relation_id': [], 
                    }
                else:
                        pesudo_samples_dict = {
                        'input': [], 'target': [], 'label_idx_cil': [], 'label_idx_til': []
                    }

                cnt_num_samples = num_samples//task_id

                prompt_for_generation = get_prompt_PCLL(text='',
                                                        label='',
                                                        gen_token=gen_token,
                                                        eos_token='', # NOTE: we do not use eos_token in the prompt
                                                        task_id=None if self.params.il_mode=='CIL' else task_id)[0]

                while cnt_num_samples > 0:
                        
                    generate_num = generate_batch_size if cnt_num_samples>=generate_batch_size else cnt_num_samples

                    # 메모리 효율적 토크나이징
                    lm_input = self.tokenizer([prompt_for_generation for _ in range(generate_num)],
                                                padding='longest',
                                                return_tensors='pt',
                                                truncation=True,
                                                max_length=256)  # 최대 길이 제한
                    lm_input = {k:v.to(model.backbone_model.device) for k,v in lm_input.items()}
                    max_input_len = np.max([len(lm_input['input_ids'][i]) for i in range(generate_num)])

                    try:
                        # 메모리 절약을 위한 generation 설정
                        with torch.no_grad():
                            generate_ids_all = model.backbone_model.generate(
                                **{'input_ids':lm_input['input_ids'], 
                                   'attention_mask':lm_input['attention_mask']}, 
                                max_new_tokens=min(50, self.params.max_seq_length-max_input_len),  # 최대 토큰 수 제한
                                pad_token_id=self.tokenizer.eos_token_id,
                                do_sample=True,
                                top_k=self.params.PCLL_KD_topk,
                                top_p=0.95,
                                temperature=1.0,
                                use_cache=True
                            ) 
                        generate_ids = generate_ids_all[:,max_input_len:].contiguous()
                        generated_samples = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

                        # 중간 텐서 정리
                        del lm_input, generate_ids_all, generate_ids
                        torch.cuda.empty_cache()

                        for _one_sample in generated_samples:
                            if _one_sample.count(' "? Answer: ')!=1:
                                continue
                            _question, _answer = _one_sample.split(' "? Answer: ')
                            _answer = _answer.replace(self.tokenizer.eos_token,'').strip()
                            pesudo_samples_dict['input'].append(_question)
                            pesudo_samples_dict['target'].append(_answer)
                            pesudo_samples_dict['label_idx_cil'].append(-1)
                            pesudo_samples_dict['label_idx_til'].append(-1)
                            if self.params.il_mode == 'IIL':
                                pesudo_samples_dict['instance_id'].append(-1)
                                pesudo_samples_dict['concept_id'].append(-1)
                                pesudo_samples_dict['relation_id'].append(-1)
                        
                        # 텍스트 데이터 정리
                        del generated_samples
                        
                    except Exception as e:
                        logger.warning(f"Generation failed for task {t_id}: {e}")
                        
                    cnt_num_samples -= generate_num
                    
                    # 배치별 메모리 정리
                    torch.cuda.empty_cache()
            
                if len(pesudo_samples_dict['input'])==0:
                    logger.info('No pseudo samples are generated in the correct format for task %d!'%(t_id+1))
                    continue
                pseudo_dataset = Dataset.from_dict(pesudo_samples_dict)
                pseudo_dataset = pseudo_dataset.map(preprocess_function_train_and_predict_generative_PCLL, 
                                                    batched=True, 
                                                    desc='Generate pseudo samples for task %d'%(t_id+1), 
                                                    batch_size=1000,
                                                    fn_kwargs={
                                                        'params':self.params,
                                                        'tokenizer':self.tokenizer,
                                                        'num_task':num_task,
                                                        'task_id':task_id,
                                                        'input_column':input_column,
                                                        'target_column':target_column,
                                                        'eos_token':eos_token,
                                                        'gen_token':gen_token,
                                                    })
                if self.params.il_mode == 'IIL':
                    pseudo_dataset.set_format(type='torch', columns=[
                        'input_ids', 'attention_mask',
                        'label_idx_cil', 'label_idx_til',
                        'input_ids_prompt', 'attention_mask_prompt',
                        'input_ids_prompt_input', 'attention_mask_prompt_input', 'labels_gen_prompt_input',
                        'input_ids_prompt_input_anstoken', 'attention_mask_prompt_input_anstoken',
                        'input_ids_prompt_input_anstoken_ans', 'attention_mask_prompt_input_anstoken_ans', 'labels_qa_prompt_input_anstoken_ans', 'labels_gen_prompt_input_anstoken_ans',
                        'target', 'instance_id', 'concept_id', 'relation_id'
                    ])
                else:
                    pseudo_dataset.set_format(type='torch', columns=[
                        'input_ids', 'attention_mask',
                        'label_idx_cil', 'label_idx_til',
                        'input_ids_prompt', 'attention_mask_prompt',
                        'input_ids_prompt_input', 'attention_mask_prompt_input', 'labels_gen_prompt_input',
                        'input_ids_prompt_input_anstoken', 'attention_mask_prompt_input_anstoken',
                        'input_ids_prompt_input_anstoken_ans', 'attention_mask_prompt_input_anstoken_ans', 'labels_qa_prompt_input_anstoken_ans', 'labels_gen_prompt_input_anstoken_ans'
                    ])

                pseudo_dataset_list.append(pseudo_dataset)

        return pseudo_dataset_list

    # ===========================================================================================

class PCLLModel(nn.Module):
    '''
        We use a wrapper for a backbone and a classifier 
        because Deepspeed only support one nn.Mudule() when calling accelerate.prepare().
    '''
    def __init__(self, model, params) -> None:
        super().__init__()
        self.backbone_model = model
        self.params = params

        self.feature_dim = self.backbone_model.module.config.hidden_size if hasattr(self.backbone_model,'module') else self.backbone_model.config.hidden_size

        # 컴포넌트들을 초기화 (device는 나중에 설정)
        self.prior_mean = Conv1D(self.feature_dim, self.feature_dim)
        self.prior_logvar = Conv1D(self.feature_dim, self.feature_dim)
        self.post_mean = Conv1D(self.feature_dim, self.feature_dim)
        self.post_logvar = Conv1D(self.feature_dim, self.feature_dim)
        self.avg_attn = AverageSelfAttention(self.feature_dim)
        self.latent_mlp = nn.Linear(self.feature_dim, self.feature_dim, bias=False)

        # Loss 함수들 (device-agnostic)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 초기화 후 backbone과 같은 device로 이동
        backbone_device = next(self.backbone_model.parameters()).device
        self._move_components_to_device(backbone_device)
    
    def _move_components_to_device(self, device):
        """모든 컴포넌트를 지정된 device로 안전하게 이동"""
        components = [
            'prior_mean', 'prior_logvar', 'post_mean', 'post_logvar', 
            'avg_attn', 'latent_mlp'
        ]
        
        for comp_name in components:
            if hasattr(self, comp_name):
                comp = getattr(self, comp_name)
                if comp is not None:
                    try:
                        setattr(self, comp_name, comp.to(device))
                    except Exception as e:
                        logger.warning(f"Failed to move {comp_name} to {device}: {e}")

    def to(self, device):
        """Override to method to ensure all components are moved to the same device (메모리 효율적)"""
        try:
            super().to(device)
            self._move_components_to_device(device)
            
            # backbone_model도 이동
            if hasattr(self, 'backbone_model') and self.backbone_model is not None:
                self.backbone_model = self.backbone_model.to(device)
                    
            # 메모리 정리
            torch.cuda.empty_cache()
            return self
        except Exception as e:
            logger.warning(f"Failed to move PCLLModel to {device}: {e}")
            return self
    
    def cpu(self):
        """Move model to CPU safely"""
        return self.to(torch.device('cpu'))

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()
        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)
    
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=0.9):
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L 

    def vae_kl_loss(self, mean1, logvar1, mean2, logvar2): # [batch_size(64), hidden_size(768)]
        # print(mean1.size(),logvar1.size(),mean2.size(),logvar2.size(),flush=True)
        exponential = logvar1 - logvar2 - \
            torch.pow(mean1 - mean2, 2) / logvar2.exp() - \
            torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()
     
    def compute_lm_loss(self, lm_input, teacher_backbone_model=None):
        # Device 처리 및 QA Loss 계산 (메모리 최적화)
        device = next(self.backbone_model.parameters()).device
        
        # 모든 입력 텐서를 적절한 device로 이동
        device_lm_input = {}
        for k, v in lm_input.items():
            if isinstance(v, torch.Tensor):
                device_lm_input[k] = v.to(device, non_blocking=True)
            else:
                device_lm_input[k] = v
        
        if teacher_backbone_model is None:
            qa_loss = self.backbone_model(**{'input_ids':device_lm_input['input_ids_prompt_input_anstoken_ans'], 
                                        'attention_mask':device_lm_input['attention_mask_prompt_input_anstoken_ans'],
                                        'labels':device_lm_input['labels_qa_prompt_input_anstoken_ans']}).loss
        else:
            with torch.no_grad():
                teacher_qa_logits = teacher_backbone_model(**{'input_ids':device_lm_input['input_ids_prompt_input_anstoken_ans'], 
                                    'attention_mask':device_lm_input['attention_mask_prompt_input_anstoken_ans']}).logits
                teacher_qa_logits_flatten = teacher_qa_logits[device_lm_input['labels_qa_prompt_input_anstoken_ans']!=-100]

            student_qa_output = self.backbone_model(**{'input_ids':device_lm_input['input_ids_prompt_input_anstoken_ans'], 
                                        'attention_mask':device_lm_input['attention_mask_prompt_input_anstoken_ans'],
                                        'labels':device_lm_input['labels_qa_prompt_input_anstoken_ans']})
            student_qa_logits = student_qa_output.logits
            student_qa_logits_flatten = student_qa_logits[device_lm_input['labels_qa_prompt_input_anstoken_ans']!=-100]
            qa_loss = student_qa_output.loss
            temp = float(self.params.PCLL_KD_temperature)
            qa_loss += self.kl_loss(F.log_softmax(student_qa_logits_flatten/temp,dim=-1),
                                        F.softmax(teacher_qa_logits_flatten/temp,dim=-1))*(temp**2)
            
            # 중간 텐서 정리
            del teacher_qa_logits, teacher_qa_logits_flatten
            del student_qa_output, student_qa_logits, student_qa_logits_flatten

        # Generation Loss 계산 (메모리 최적화)
        if teacher_backbone_model is None:
            generation_loss = self.backbone_model(**{'input_ids':device_lm_input['input_ids_prompt_input_anstoken_ans'], 
                                    'attention_mask':device_lm_input['attention_mask_prompt_input_anstoken_ans'],
                                    'labels':device_lm_input['labels_gen_prompt_input_anstoken_ans']}).loss
        else:
            with torch.no_grad():
                teacher_gen_logits = teacher_backbone_model(**{'input_ids':device_lm_input['input_ids_prompt_input_anstoken_ans'], 
                                    'attention_mask':device_lm_input['attention_mask_prompt_input_anstoken_ans']}).logits
                teacher_gen_logits_flatten = teacher_gen_logits[device_lm_input['labels_gen_prompt_input_anstoken_ans']!=-100]

            student_gen_output = self.backbone_model(**{'input_ids':device_lm_input['input_ids_prompt_input_anstoken_ans'], 
                                        'attention_mask':device_lm_input['attention_mask_prompt_input_anstoken_ans'],
                                        'labels':device_lm_input['labels_gen_prompt_input_anstoken_ans']})
            student_gen_logits = student_gen_output.logits
            student_gen_logits_flatten = student_gen_logits[device_lm_input['labels_gen_prompt_input_anstoken_ans']!=-100]
            generation_loss = student_gen_output.loss
            temp = float(self.params.PCLL_KD_temperature)
            generation_loss += self.kl_loss(F.log_softmax(student_gen_logits_flatten/temp,dim=-1),
                                        F.softmax(teacher_gen_logits_flatten/temp,dim=-1))*(temp**2)
            
            # 중간 텐서 정리
            del teacher_gen_logits, teacher_gen_logits_flatten
            del student_gen_output, student_gen_logits, student_gen_logits_flatten

        result = qa_loss + self.params.PCLL_weight_gen_loss*generation_loss
        
        # 최종 정리
        del qa_loss, generation_loss, device_lm_input
        
        return result

    def compute_vae_loss(self, lm_input, teacher_backbone_model=None, beta=0):
        # * Get posterior: latent representation p(z|x,p) (메모리 최적화)
        # Device를 backbone_model의 device로 결정
        device = next(self.backbone_model.parameters()).device
        
        # 모든 입력 텐서를 적절한 device로 이동
        device_lm_input = {}
        for k, v in lm_input.items():
            if isinstance(v, torch.Tensor):
                device_lm_input[k] = v.to(device, non_blocking=True)
            else:
                device_lm_input[k] = v
        
        # 모든 컴포넌트가 같은 device에 있도록 보장
        components_to_move = ['post_mean', 'post_logvar', 'prior_mean', 'prior_logvar', 'latent_mlp', 'avg_attn']
        for comp_name in components_to_move:
            if hasattr(self, comp_name):
                comp = getattr(self, comp_name)
                if comp is not None:
                    setattr(self, comp_name, comp.to(device))

        # Posterior computation
        post_out = self.backbone_model(**{'input_ids':device_lm_input['input_ids_prompt_input'], 
                                 'attention_mask':device_lm_input['attention_mask_prompt_input']},
                                 output_hidden_states=True).hidden_states[-1]
        post_emb, _ = self.avg_attn(post_out, attention_mask=device_lm_input['attention_mask_prompt_input'])
        posterior_mean, posterior_logvar = self.post_mean(post_emb), self.post_logvar(post_emb)
        
        # 중간 텐서 정리
        del post_out, post_emb

        # Prior computation  
        prior_out = self.backbone_model(**{'input_ids':device_lm_input['input_ids_prompt'], 
                                 'attention_mask':device_lm_input['attention_mask_prompt']},
                                 output_hidden_states=True).hidden_states[-1]
        prior_emb, _ = self.avg_attn(prior_out, attention_mask=device_lm_input['attention_mask_prompt'])
        prior_mean, prior_logvar = self.prior_mean(prior_emb), self.prior_logvar(prior_emb)
        
        # 중간 텐서 정리
        del prior_out, prior_emb

        # Latent variable computation
        latent_z = self.reparameterize(posterior_mean, posterior_logvar)
        latent_proj = self.params.PCLL_alpha_z * self.latent_mlp(latent_z)
        assert not torch.isnan(latent_z).any(), 'Training gets NAN z!'

        kl_loss = self.vae_kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar).unsqueeze(0)
        
        # 중간 텐서 정리
        del posterior_mean, posterior_logvar, prior_mean, prior_logvar, latent_z
        
        # Get input embeddings
        input_embeddings = self.backbone_model.get_input_embeddings()
        input_embed = input_embeddings(device_lm_input['input_ids_prompt_input'])
        
        if teacher_backbone_model is None:
            reconstruct_loss = self.backbone_model(**{'inputs_embeds':input_embed+latent_proj, 
                                    'attention_mask':device_lm_input['attention_mask_prompt_input'],
                                    'labels':device_lm_input['labels_gen_prompt_input']}).loss
        else:
            with torch.no_grad():
                teacher_reconstruct_logits = teacher_backbone_model(**{'inputs_embeds':input_embed+latent_proj, 
                                    'attention_mask':device_lm_input['attention_mask_prompt_input']}).logits
                teacher_reconstruct_logits_flatten = teacher_reconstruct_logits[device_lm_input['labels_gen_prompt_input']!=-100]
                
            generation_output = self.backbone_model(**{'inputs_embeds':input_embed+latent_proj, 
                                    'attention_mask':device_lm_input['attention_mask_prompt_input'],
                                    'labels':device_lm_input['labels_gen_prompt_input']})
            
            reconstruct_loss = generation_output.loss
            reconstruct_logits = generation_output.logits
            reconstruct_logits_flatten = reconstruct_logits[device_lm_input['labels_gen_prompt_input']!=-100]

            temp = float(self.params.PCLL_KD_temperature)
            reconstruct_loss += self.kl_loss(F.log_softmax(reconstruct_logits_flatten/temp,dim=-1),
                                        F.softmax(teacher_reconstruct_logits_flatten/temp,dim=-1))*(temp**2)
            
            # 중간 텐서 정리
            del teacher_reconstruct_logits, teacher_reconstruct_logits_flatten
            del generation_output, reconstruct_logits, reconstruct_logits_flatten
        
        # 중간 텐서 정리
        del input_embed, latent_proj
        
        result = reconstruct_loss + beta*0.01*kl_loss
        
        # 최종 정리
        del reconstruct_loss, kl_loss
        
        return result

class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = gelu

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        # Ensure attention_weights is on the same device as inputs
        attention_weights = self.attention_weights.to(inputs.device)
        scores = self.non_linearity(inputs.matmul(attention_weights))

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        if attention_mask is not None:
            scores = scores + attention_mask

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        # print('weighted shape',weighted.shape, flush=True)
        if len(weighted.shape) == 2:
            representations = weighted.sum(0).unsqueeze(0)
        else:
            representations = weighted.sum(1).squeeze(1)
        # print('after weighted shape',weighted.shape, flush=True) 

        return representations, scores