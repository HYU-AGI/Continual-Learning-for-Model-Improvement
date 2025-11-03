import os
import torch
import torch.nn as nn
from torch.optim import Adam
from utils.evaluation import evaluate_sent_level_acc_with_generation

def train_lm_head_cil(params, CL_dataset, train_loader_list, model, tokenizer, accelerator):
    """
    CIL 시나리오에서 사전학습된 backbone을 고정하고
    LM Head만 fine-tuning하여 자연어 클래스 라벨을 학습하는 함수

    Args:
        params: argparse.Namespace, 학습 파라미터
        CL_dataset: Continual Dataset 인스턴스 (continual_config 포함)
        train_loader_list: list of DataLoader, 각 task별 학습 데이터 로더
        model: Huggingface Transformer 모델
        tokenizer: Huggingface Tokenizer
        accelerator: Huggingface Accelerator
    Returns:
        None (학습된 모델 파라미터는 model에 반영)
    """

    # 1) Freeze backbone, unfreeze only LM Head
    for name, param in model.named_parameters():
        if 'lm_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 2) Optimizer for LM Head
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params.lm_head_lr
    )

    # 3) Prepare with Accelerator
    model.train()
    model, optimizer, *loader_list = accelerator.prepare(model, optimizer, *train_loader_list)

    # 4) Task 순서대로 LM Head만 학습
    for task_id, train_loader in enumerate(loader_list):
        if accelerator.is_main_process:
            print(f"=== Task {task_id}: LM Head CIL Training ===")
        for epoch in range(params.lm_head_epochs):
            for step, lm_input in enumerate(train_loader):
                outputs = model(
                    input_ids=lm_input['input_ids_with_ans'],
                    attention_mask=lm_input['attention_mask_with_ans'],
                    labels=lm_input['labels_with_ans']
                )
                loss = outputs.loss

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                if step % params.info_per_steps == 0 and accelerator.is_main_process:
                    print(f"Task {task_id} Epoch {epoch} Step {step}: loss={loss.item():.4f}")

        # Task 끝날 때마다 평가 (옵션)
        if accelerator.is_main_process and params.evaluate_after_task:
            acc = evaluate_sent_level_acc_with_generation(
                model=model,
                eval_data_loader=loader_list[task_id],
                tokenizer=tokenizer,
                accelerator=accelerator,
                params=params,
                idx2label=CL_dataset.continual_config['idx2label']
            )
            print(f"-- After Task {task_id} Eval Acc: {acc:.2f}%")

    # 5) 학습된 LM Head 저장
    if params.save_lm_head:
        os.makedirs(params.dump_path, exist_ok=True)
        save_path = os.path.join(params.dump_path, 'lm_head_cil.pt')
        torch.save(model.lm_head.state_dict(), save_path)
        if accelerator.is_main_process:
            print(f"Saved LM Head weights to {save_path}")