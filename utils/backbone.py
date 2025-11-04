
import os
import logging
import torch
from adapters import AutoAdapterModel
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import (
    LoraConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)

from utils.prompt import get_auto_prompt_tuning_init_text

# -----------------------------------------------------------------------------
#  지원하는 백본 이름 → 타입 매핑
# -----------------------------------------------------------------------------
BACKBONE2TYPE = {
    # ── Generative ───────────────────────────────────────────────
    "gpt2": "generative",
    "gpt2-large": "generative",
    "EleutherAI/pythia-70m-deduped": "generative",
    "EleutherAI/pythia-160m-deduped": "generative",
    "EleutherAI/pythia-410m-deduped": "generative",
    "EleutherAI/pythia-1b-deduped": "generative",
    "EleutherAI/pythia-1.4b-deduped": "generative",
    "EleutherAI/pythia-2.8b-deduped": "generative",
    "baffo32/decapoda-research-llama-7B-hf": "generative",
    "meta-llama/Llama-3.2-1B": "generative",
    "lmsys/vicuna-7b-v1.1": "generative",
    "llama2-13b-orca-8k-3319": "generative",
    # Qwen local checkpoints
    "Qwen3-0.6B": "generative",
    "Qwen2-0.5B": "generative",
    "Qwen2.5-0.5B": "generative",
    "gpt-oss-20b": "generative",
    # ── Discriminative ────────────────────────────────────────────
    "roberta-base": "discriminative",
    "roberta-large": "discriminative",
    "bert-base-cased": "discriminative",
    "bert-base-uncased": "discriminative",
    "bert-large-cased": "discriminative",
    "bert-large-uncased": "discriminative",
}

# -----------------------------------------------------------------------------
#  공통 유틸: 토크나이저 & 임베딩 동기화
# -----------------------------------------------------------------------------

def _sync_tokenizer_and_embeddings(tokenizer, model, params, num_task: int = 1):
    """Ensure pad/special tokens exist and resize model embeddings if needed."""
    import torch
    import logging
    
    # 1) pad token 확보 ---------------------------------------------------------
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id

    # 2) 추가/특수 토큰 정의 -----------------------------------------------------
    add_tokens = ["__ans__"]
    if getattr(params, "LAMOL_use_task_specific_gen_token", True):
        add_tokens += [f"__{i}__" for i in range(num_task)]
    else:
        add_tokens.append("__gen__")

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})

    # 3) 임베딩 리사이즈 ---------------------------------------------------------
    if num_added > 0:
        try:
            # Meta device에서 mean_resizing 비활성화하여 linalg_eig 회피
            embed_tokens = model.get_input_embeddings()
            lm_head = model.get_output_embeddings()
            
            # Meta device 확인
            if embed_tokens.weight.device.type == 'meta':
                logging.warning("[Backbone] Meta device detected. Using safe resize method.")
                # mean_resizing=False로 설정하여 고유값 분해 연산 회피
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            else:
                model.resize_token_embeddings(len(tokenizer))
                
            model.config.vocab_size = len(tokenizer)  # generate 안전성 확보
            logging.info(
                f"[Backbone] Added {num_added} tokens → new vocab_size = {len(tokenizer)}"
            )
        except Exception as e:
            logging.error(f"[Backbone] Failed to resize embeddings: {e}")
            
            # Meta device에서는 임베딩 리사이즈를 건너뛰고 config만 업데이트
            if embed_tokens.weight.device.type == 'meta':
                logging.warning("[Backbone] Skipping embedding resize for Meta device. Config updated only.")
                model.config.vocab_size = len(tokenizer)
                # tokenizer는 새 토큰이 추가된 상태이므로 그대로 사용
                logging.info(f"[Backbone] Config updated: vocab_size = {len(tokenizer)}")
            else:
                # 실제 device에서는 수동으로 임베딩 크기 조정
                old_vocab_size = embed_tokens.weight.shape[0]
                new_vocab_size = len(tokenizer)
                
                # 새로운 임베딩 가중치 생성
                embed_dim = embed_tokens.weight.shape[1]
                
                # CPU에서 새 가중치 생성
                new_embed_weight = torch.zeros(new_vocab_size, embed_dim)
                new_embed_weight[:old_vocab_size] = embed_tokens.weight.to('cpu')
                
                # 새로운 토큰은 정규분포로 초기화
                if new_vocab_size > old_vocab_size:
                    std = embed_tokens.weight.std().item()
                    new_embed_weight[old_vocab_size:].normal_(mean=0.0, std=std)
                
                # 임베딩 교체
                embed_tokens.weight = torch.nn.Parameter(new_embed_weight)
                
                # lm_head가 별도인 경우 처리
                if lm_head is not embed_tokens:
                    new_lm_weight = torch.zeros(new_vocab_size, embed_dim)
                    new_lm_weight[:old_vocab_size] = lm_head.weight.to('cpu')
                    if new_vocab_size > old_vocab_size:
                        new_lm_weight[old_vocab_size:].normal_(mean=0.0, std=std)
                    lm_head.weight = torch.nn.Parameter(new_lm_weight)
                
                model.config.vocab_size = new_vocab_size
                logging.info(f"[Backbone] Manual resize completed: {old_vocab_size} → {new_vocab_size}")


# -----------------------------------------------------------------------------
#  메인 빌더 함수
# -----------------------------------------------------------------------------

def get_backbone(params, num_task: int = 1):
    """load backbone model & tokenizer and apply PEFT if needed"""

    # ───────────────────── 백본 타입 결정 ────────────────────────────
    if params.backbone_type == "auto":
        assert (
            params.backbone in BACKBONE2TYPE
        ), f"Not implemented for backbone {params.backbone}"
        setattr(params, "backbone_type", BACKBONE2TYPE[params.backbone])
    else:
        assert params.backbone_type in (
            "generative",
            "discriminative",
        ), f"Invalid backbone_type {params.backbone_type}"

    # ======================================================================
    #  1) Llama 로컬 ckpt (Llama-3.1-8B, Llama-3.2-1B, Llama-3.2-3B)
    # ======================================================================
    if params.backbone in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]:
        model_path = f"./{params.backbone.split('/')[-1]}"
        print(model_path)

        # tokenizer 먼저 로드하여 특수 토큰 추가
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            padding_side="left" if params.backbone_type == "generative" else "right",
        )
        
        # 특수 토큰 미리 추가
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        add_tokens = ["__ans__"]
        if getattr(params, "LAMOL_use_task_specific_gen_token", True):
            add_tokens += [f"__{i}__" for i in range(num_task)]
        else:
            add_tokens.append("__gen__")
        
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
        
        # config 로드 (원래 vocab_size 유지)
        config = AutoConfig.from_pretrained(model_path)
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            if isinstance(config.rope_scaling, dict) and "rope_type" in config.rope_scaling:
                factor = config.rope_scaling.get("factor", 1.0)
                config.rope_scaling = {"type": "linear", "factor": factor}
        
        # 모델 로드 (극단적 메모리 절약)
        try:
            # 8bit quantization 시도
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=False,
                quantization_config=quantization_config,
                device_map="auto",
            )
            print("Loaded with 8bit quantization")
        except Exception as e:
            print(f"8bit quantization failed: {e}")
            try:
                # 4bit quantization 시도
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    low_cpu_mem_usage=True, 
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=False,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                print("Loaded with 4bit quantization")
            except Exception as e2:
                print(f"4bit quantization also failed: {e2}")
                # 최후의 수단: CPU 로딩
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    low_cpu_mem_usage=True, 
                    torch_dtype=torch.float32,  # CPU에서는 float32 사용
                    trust_remote_code=False,
                )
                print("Loaded on CPU with float32")
        
        # 임베딩 리사이즈 (안전한 방법으로)
        if num_added > 0:
            try:
                # Mean resizing을 비활성화하여 linalg_eig 연산 회피
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
                model.config.vocab_size = len(tokenizer)
                logging.info(f"[Backbone] Resized embeddings: +{num_added} tokens → {len(tokenizer)}")
            except Exception as e:
                logging.warning(f"[Backbone] Embedding resize failed: {e}")
                # Meta device에서는 config만 업데이트
                model.config.vocab_size = len(tokenizer)
                logging.info(f"[Backbone] Config updated: vocab_size = {len(tokenizer)}")

        # ───────────────────── PEFT (선택) ─────────────────────
        if (
            hasattr(params, "PEFT_type")
            and params.PEFT_type is not None
            and params.PEFT_type != "None"
        ):
            if params.PEFT_type == "PromptTuning":
                if params.PEFT_prompt_tuning_init_text and params.PEFT_prompt_tuning_init_text != "":
                    prompt_text = (
                        get_auto_prompt_tuning_init_text(dataset=params.dataset)
                        if params.PEFT_prompt_tuning_init_text == "auto"
                        else params.PEFT_prompt_tuning_init_text
                    )
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        prompt_tuning_init_text=prompt_text,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
            elif params.PEFT_type == "LoRA":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=params.PEFT_lora_target_modules,
                    r=params.PEFT_lora_r,
                    lora_alpha=params.PEFT_lora_alpha,
                    bias=params.PEFT_lora_bias,
                    lora_dropout=params.PEFT_lora_dropout,
                )
            else:
                raise NotImplementedError()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    # ======================================================================
    #  2) GPT-OSS-20B 로컬 체크포인트
    # ======================================================================
    elif params.backbone == "gpt-oss-20b":
        print(f"Loading gpt-oss-20b from local path: {params.backbone}")
        model_path = "./gpt-oss-20b"
        
        # tokenizer 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 특별 토큰 추가
        add_tokens = ["__ans__"]
        if getattr(params, "LAMOL_use_task_specific_gen_token", True):
            add_tokens += [f"__{i}__" for i in range(num_task)]
        else:
            add_tokens.append("__gen__")
        
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
        
        # Gradient checkpointing 활성화 (메모리 절약)
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            
        # 추가 메모리 최적화 설정
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False  # KV 캐시 비활성화로 메모리 절약
            
        # MoE 모델의 경우 추가 최적화
        if hasattr(model.config, 'num_local_experts'):
            # expert parallelism 비활성화하여 메모리 절약
            if hasattr(model.config, 'output_router_logits'):
                model.config.output_router_logits = False
        
        # 임베딩 리사이즈 (안전한 방법으로)
        if num_added > 0:
            try:
                # Mean resizing을 비활성화하여 linalg_eig 연산 회피
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
                model.config.vocab_size = len(tokenizer)
                logging.info(f"[Backbone] Resized embeddings: +{num_added} tokens → {len(tokenizer)}")
            except Exception as e:
                logging.warning(f"[Backbone] Embedding resize failed: {e}")
                # Meta device에서는 config만 업데이트
                model.config.vocab_size = len(tokenizer)
                logging.info(f"[Backbone] Config updated: vocab_size = {len(tokenizer)}")
        
        # ───────────────────── PEFT (선택) ─────────────────────
        if (
            hasattr(params, "PEFT_type")
            and params.PEFT_type is not None
            and params.PEFT_type != "None"
        ):
            if params.PEFT_type == "PromptTuning":
                if params.PEFT_prompt_tuning_init_text and params.PEFT_prompt_tuning_init_text != "":
                    prompt_text = (
                        get_auto_prompt_tuning_init_text(dataset=params.dataset)
                        if params.PEFT_prompt_tuning_init_text == "auto"
                        else params.PEFT_prompt_tuning_init_text
                    )
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        prompt_tuning_init_text=prompt_text,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
            elif params.PEFT_type == "LoRA":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=params.PEFT_lora_target_modules,
                    r=params.PEFT_lora_r,
                    lora_alpha=params.PEFT_lora_alpha,
                    bias=params.PEFT_lora_bias,
                    lora_dropout=params.PEFT_lora_dropout,
                )
            else:
                raise NotImplementedError()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    # ======================================================================
    #  3) Qwen 로컬 체크포인트 (0.5B / 0.6B)
    # ======================================================================
    elif params.backbone in ["Qwen3-0.6B", "Qwen2-0.5B", "Qwen2.5-0.5B"]:
        model_path = f"./{params.backbone}"
        print(model_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            padding_side="left" if params.backbone_type == "generative" else "right",
        )

        if (
            hasattr(params, "PEFT_type")
            and params.PEFT_type is not None
            and params.PEFT_type != "None"
        ):
            if params.PEFT_type == "PromptTuning":
                if params.PEFT_prompt_tuning_init_text and params.PEFT_prompt_tuning_init_text != "":
                    prompt_text = (
                        get_auto_prompt_tuning_init_text(dataset=params.dataset)
                        if params.PEFT_prompt_tuning_init_text == "auto"
                        else params.PEFT_prompt_tuning_init_text
                    )
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        prompt_tuning_init_text=prompt_text,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
            elif params.PEFT_type == "LoRA":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=params.PEFT_lora_target_modules,
                    r=params.PEFT_lora_r,
                    lora_alpha=params.PEFT_lora_alpha,
                    bias=params.PEFT_lora_bias,
                    lora_dropout=params.PEFT_lora_dropout,
                )
            else:
                raise NotImplementedError()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    # ======================================================================
    #  4) 그 외 HuggingFace 허브 모델 (온라인/캐시)
    # ======================================================================
    else:
        try:
            config = AutoConfig.from_pretrained(params.backbone)
            config.return_dict = True
        except Exception:
            config = AutoConfig.from_pretrained(
                os.path.join(params.backbone_cache_path, params.backbone)
            )
            config.return_dict = True

        if params.method == "AdapterCL":
            model = AutoAdapterModel.from_pretrained(params.backbone, config=config)

        if params.method == "CPFD":
            config.output_attentions = True

        # ── 모델 로드 ---------------------------------------------------------
        def _load_model_from_pretrained(path, **kwargs):
            return AutoModelForCausalLM.from_pretrained(
                path, low_cpu_mem_usage=True, device_map="auto", **kwargs
            )

        if params.backbone_revision:
            # 사용자가 revision을 지정한 경우
            cache_dir = os.path.join(
                params.backbone_cache_path,
                os.path.join(os.path.basename(params.backbone), params.backbone_revision),
            )
            try:
                model = _load_model_from_pretrained(
                    params.backbone,
                    config=config,
                    revision=params.backbone_revision,
                    cache_dir=cache_dir,
                )
            except Exception:
                model = _load_model_from_pretrained(
                    os.path.join(cache_dir), config=config
                )
        else:
            # 기본 최신 revision 사용
            try:
                model = _load_model_from_pretrained(params.backbone, config=config)
            except Exception:
                model = _load_model_from_pretrained(
                    os.path.join(params.backbone_cache_path, params.backbone),
                    config=config,
                )

        # 무작위 초기화 옵션
        if getattr(params, "backbone_random_init", False):
            model.apply(model._init_weights)

        # ── PEFT --------------------------------------------------------------
        if (
            hasattr(params, "PEFT_type")
            and params.PEFT_type is not None
            and params.PEFT_type != "None"
        ):
            if params.PEFT_type == "PromptTuning":
                if params.PEFT_prompt_tuning_init_text and params.PEFT_prompt_tuning_init_text != "":
                    prompt_text = (
                        get_auto_prompt_tuning_init_text(dataset=params.dataset)
                        if params.PEFT_prompt_tuning_init_text == "auto"
                        else params.PEFT_prompt_tuning_init_text
                    )
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        prompt_tuning_init_text=prompt_text,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
            elif params.PEFT_type == "LoRA":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=params.PEFT_lora_target_modules,
                    r=params.PEFT_lora_r,
                    lora_alpha=params.PEFT_lora_alpha,
                    bias=params.PEFT_lora_bias,
                    lora_dropout=params.PEFT_lora_dropout,
                )
            else:
                raise NotImplementedError()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # ── 토크나이저 로드 ----------------------------------------------------
        def _load_tokenizer(path, **kwargs):
            return AutoTokenizer.from_pretrained(
                path,
                padding_side="left" if params.backbone_type == "generative" else "right",
                **kwargs,
            )

        if params.backbone_revision:
            cache_dir = os.path.join(
                params.backbone_cache_path,
                os.path.join(os.path.basename(params.backbone), params.backbone_revision),
            )
            try:
                tokenizer = _load_tokenizer(
                    params.backbone, revision=params.backbone_revision, cache_dir=cache_dir
                )
            except Exception:
                tokenizer = _load_tokenizer(os.path.join(cache_dir))
        else:
            try:
                tokenizer = _load_tokenizer(params.backbone)
            except Exception:
                tokenizer = _load_tokenizer(
                    os.path.join(params.backbone_cache_path, params.backbone)
                )

    # ======================================================================
    #  공통 후처리 (pad_token·special_token·임베딩 동기화)
    # ======================================================================
    # Llama 및 GPT-OSS-20B 모델들은 이미 위에서 처리했으므로 건너뛰기
    if params.backbone_type == "generative" and params.backbone not in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "gpt-oss-20b"]:
        _sync_tokenizer_and_embeddings(tokenizer, model, params, num_task)

    # pad_token 안전장치 (discriminative 포함)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if hasattr(model, "config") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    print(model)
    return model, tokenizer

# -----------------------------------------------------------------------------
#  obtain_features / obtain_generate_ids: 기존 코드 그대로 (생략 없음)
# -----------------------------------------------------------------------------

def _validate_lm_input_for_features(lm_input, model, context="obtain_features"):
    """obtain_features용 안전 검증"""
    logger = logging.getLogger()
    validated_input = {}
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if "input_ids" in lm_input:
        input_ids = lm_input["input_ids"]
        if input_ids.max() >= vocab_size or input_ids.min() < 0:
            logger.warning(
                f"[{context}] input_ids out of range (0–{vocab_size-1}). Clamping to unk_token."
            )
            unk_id = getattr(model.config, "unk_token_id", 0) or 0
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        validated_input["input_ids"] = input_ids
    if "attention_mask" in lm_input:
        validated_input["attention_mask"] = lm_input["attention_mask"]
    for k in lm_input:
        if k not in validated_input:
            validated_input[k] = lm_input[k]
    return validated_input


def obtain_features(params, model, lm_input, tokenizer):
    """Extract last hidden state features for classification"""
    safe_input = _validate_lm_input_for_features(lm_input, model)
    if params.backbone_type == "generative":
        assert params.classification_type == "sentence-level"
        all_hs = model(
            input_ids=safe_input["input_ids"],
            attention_mask=safe_input["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        ).hidden_states
        assert params.backbone_extract_token == "last_token"
        feat = all_hs[-1][:, -1, :].contiguous()
    elif params.backbone_type == "discriminative":
        all_hs = model(
            input_ids=safe_input["input_ids"],
            attention_mask=safe_input["attention_mask"],
            output_hidden_states=True,
        ).hidden_states
        if params.classification_type == "sentence-level":
            if params.backbone_extract_token == "last_token":
                idx = safe_input["attention_mask"].sum(dim=-1) - 1
                last = all_hs[-1]
                batch, seq, dim = last.size()
                idx = idx.view(-1, 1, 1).expand(-1, 1, dim)
                feat = last.gather(1, idx).squeeze(1).contiguous()
            elif params.backbone_extract_token == "cls_token":
                feat = all_hs[-1][:, 0, :].contiguous()
            else:
                raise NotImplementedError()
        elif params.classification_type == "word-level":
            feat = all_hs[-1]
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return feat


def obtain_generate_ids(params, model, lm_input, tokenizer):
    """Generate continuation ids"""
    safe_input = _validate_lm_input_for_features(lm_input, model, "obtain_generate_ids")
    in_len = safe_input["input_ids"].shape[1]
    gen_all = model.generate(
        input_ids=safe_input["input_ids"],
        attention_mask=safe_input["attention_mask"],
        max_new_tokens=params.backbone_max_new_token,
        pad_token_id=tokenizer.eos_token_id,
    )
    return gen_all[:, in_len:].contiguous()