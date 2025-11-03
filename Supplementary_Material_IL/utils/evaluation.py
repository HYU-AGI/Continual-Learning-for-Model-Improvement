import logging
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score
from random import shuffle

from utils.backbone import obtain_features, obtain_generate_ids
from utils.prompt import get_prompt_ICL

logger = logging.getLogger()

# ====================== the evaluation metric for each task ==========================================================================

# ------------------------------------- sentence level evaluation ----------------------------------------------
# def evaluate_sent_level_acc_with_generation(model, eval_data_loader, tokenizer, accelerator, params, idx2label):
#     '''
#         Evaluate the accuracy using generation for sentence-level classification tasks

#         Return:
#          - Acc (%)
#     '''
#     model.eval()
#     acc_list_all = []
#     save_dict = {}
    
#     with torch.no_grad():
#         for lm_input in eval_data_loader: 

#             label_idx = lm_input['label_idx_cil']
#             generate_ids = obtain_generate_ids(params=params, 
#                                                 model=model, 
#                                                 lm_input=lm_input, 
#                                                 tokenizer=tokenizer)

#             # Gather from different processes in DDP
#             generate_ids = accelerator.pad_across_processes(
#                                                 generate_ids, 
#                                                 dim=1, 
#                                                 pad_index=tokenizer.eos_token_id, 
#                                                 pad_first=False)
#             generate_ids, label_idx = accelerator.gather_for_metrics((generate_ids, label_idx))

#             generate_seq = tokenizer.batch_decode(generate_ids,skip_special_tokens=True)
#             generate_seq = [pred.strip(' \n').lower().split(tokenizer.eos_token)[0] for pred in generate_seq]

#             # Compare the target directly
#             if idx2label == []:
#                 target_output = lm_input['target']

#                 acc_list = [1 if pred.strip().lower()==target.strip().lower() else 0 for pred, target in zip(generate_seq,target_output)]
#                 acc_list_all.extend(acc_list)

#                 if params.il_mode == 'IIL':
#                     for i, (pred, target) in enumerate(zip(generate_seq,target_output)):
#                         save_dict[lm_input['instance_id'][i].item()] = {
#                             'instance_id': lm_input['instance_id'][i].item(),
#                             'relation_id': lm_input['relation_id'][i].item(),
#                             'concept_id': lm_input['concept_id'][i].item(),
#                             'prediton': pred.strip().lower(),
#                             'target': target.strip().lower(),
#                         }

#             # Compare the class idx
#             else:
#                 label_idx = label_idx.detach().cpu().numpy()
#                 target_output = [idx2label[l] for l in label_idx]
#                 target_output = [target.strip(' \n').lower() for target in target_output]

#                 acc_list = [1 if pred==target else 0 for pred, target in zip(generate_seq,target_output)]
#                 acc_list_all.extend(acc_list)
    
#     acc = np.round(np.mean(acc_list_all)*100,3)
#     model.train()

#     if params.il_mode == 'IIL':
#         return acc, save_dict

#     return acc

def evaluate_sent_level_acc_with_generation(model,
                                            eval_data_loader,
                                            next_eval_data_loader,
                                            tokenizer,
                                            accelerator,
                                            params,
                                            idx2label):
    """
    Evaluate the accuracy using generation for sentence-level classification tasks.

    This function evaluates both the current task and the next task (if provided),
    returning their accuracies.

    Returns:
     - acc (float): accuracy for current task (%)
     - acc_next (float or None): accuracy for next task (%) or None if not evaluated
    """
    model.eval()
    acc_list_all = []
    acc_list_next = []

    print(f"eval_data_loader: {len(eval_data_loader)}")
    if next_eval_data_loader is not None:
        print(f"next_eval_data_loader: {len(next_eval_data_loader)}")

    # --- Current task evaluation ---
    with torch.no_grad():
        for lm_input in eval_data_loader:
            # 1) Generation
            generate_ids = obtain_generate_ids(
                params=params, model=model, lm_input=lm_input, tokenizer=tokenizer
            )
            # 2) DDP pad & gather
            generate_ids = accelerator.pad_across_processes(
                generate_ids, dim=1,
                pad_index=tokenizer.eos_token_id,
                pad_first=False
            )
            label_idx = lm_input['label_idx_cil']
            generate_ids, label_idx = accelerator.gather_for_metrics((generate_ids, label_idx))

            # 3) Decode
            generate_seq = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            generate_seq = [
                pred.strip(' \n').lower().split(tokenizer.eos_token)[0]
                for pred in generate_seq
            ]

            # 4) Compare with targets
            if idx2label == []:
                # raw string targets
                target_output = lm_input['target']
                for pred, target in zip(generate_seq, target_output):
                    acc_list_all.append(1 if pred.strip().lower() == target.strip().lower() else 0)
            else:
                # class-index targets
                label_np = label_idx.detach().cpu().numpy()
                target_output = [idx2label[l] for l in label_np]
                target_output = [t.strip(' \n').lower() for t in target_output]
                for pred, target in zip(generate_seq, target_output):
                    acc_list_all.append(1 if pred == target else 0)

    acc = np.round(np.mean(acc_list_all) * 100, 3)

    # --- Next task evaluation (if provided) ---
    if next_eval_data_loader is not None:
        with torch.no_grad():
            for lm_input in next_eval_data_loader:
                generate_ids = obtain_generate_ids(
                    params=params, model=model, lm_input=lm_input, tokenizer=tokenizer
                )
                generate_ids = accelerator.pad_across_processes(
                    generate_ids, dim=1,
                    pad_index=tokenizer.eos_token_id,
                    pad_first=False
                )
                label_idx = lm_input['label_idx_cil']
                generate_ids, label_idx = accelerator.gather_for_metrics((generate_ids, label_idx))

                generate_seq = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
                generate_seq = [
                    pred.strip(' \n').lower().split(tokenizer.eos_token)[0]
                    for pred in generate_seq
                ]

                if idx2label == []:
                    target_output = lm_input['target']
                    for pred, target in zip(generate_seq, target_output):
                        acc_list_next.append(1 if pred.strip().lower() == target.strip().lower() else 0)
                else:
                    label_np = label_idx.detach().cpu().numpy()
                    target_output = [idx2label[l] for l in label_np]
                    target_output = [t.strip(' \n').lower() for t in target_output]
                    for pred, target in zip(generate_seq, target_output):
                        acc_list_next.append(1 if pred == target else 0)

        acc_next = np.round(np.mean(acc_list_next) * 100, 3)
    else:
        acc_next = None

    model.train()
    return acc, acc_next


def evaluate_sent_level_acc_with_classifier(model, classifier_list, cur_task_id,
                                            eval_data_loader, next_eval_data_loader,
                                            tokenizer, accelerator, params, idx2label):
    """
    Evaluate the accuracy with classifier for sentence-level classification tasks.

    This function evaluates both the current task (cur_task_id) and the next task (cur_task_id+1),
    using the appropriate data for each task, and returning both accuracies separately.

    Return:
     - acc (float): accuracy for cur_task_id
     - acc_next_task (float or None): accuracy for cur_task_id+1
    """

    model.eval()

    acc_list_all = []
    acc_list_next = []

    print(f"eval_data_loader: {len(eval_data_loader)}")
    if next_eval_data_loader is not None:
        print(f"next_eval_data_loader: {len(next_eval_data_loader)}")

    # Evaluate current task
    with torch.no_grad():
        for lm_input in eval_data_loader:
            extracted_features = obtain_features(
                params=params,
                model=model,
                lm_input=lm_input,
                tokenizer=tokenizer
            )
            # Determine feature device (handles model sharded across GPUs)
            feat_device = extracted_features.device

            if params.il_mode == 'CIL':
                logits = torch.cat([
                    classifier_list[i].to(feat_device)(extracted_features)
                    for i in range(cur_task_id + 1)
                ], dim=-1)
                preds = logits.argmax(dim=-1)
                label_idx = lm_input['label_idx_cil']
            elif params.il_mode == 'TIL':
                classifier = classifier_list[cur_task_id].to(feat_device)
                logits = classifier(extracted_features)
                preds = logits.argmax(dim=-1)
                label_idx = lm_input['label_idx_til']
            else:
                raise NotImplementedError(f"Unknown IL mode {params.il_mode}")

            preds, label_idx = accelerator.gather_for_metrics((preds, label_idx))
            label_idx = label_idx.detach().cpu().numpy()
            acc_list_all.extend([1 if p == t else 0 for p, t in zip(preds, label_idx)])

    acc = np.round(np.mean(acc_list_all) * 100, 3)

    # Evaluate next task (if exists)
    if cur_task_id + 1 < len(classifier_list) and next_eval_data_loader is not None:
        with torch.no_grad():
            for lm_input in next_eval_data_loader:
                extracted_features = obtain_features(
                    params=params,
                    model=model,
                    lm_input=lm_input,
                    tokenizer=tokenizer
                )
                feat_device = extracted_features.device

                if params.il_mode == 'CIL':
                    logits_next = torch.cat([
                        classifier_list[i].to(feat_device)(extracted_features)
                        for i in range(cur_task_id + 2)
                    ], dim=-1)
                    preds_next = logits_next.argmax(dim=-1)
                    label_idx_next = lm_input['label_idx_cil']
                elif params.il_mode == 'TIL':
                    classifier_next = classifier_list[cur_task_id + 1].to(feat_device)
                    logits_next = classifier_next(extracted_features)
                    preds_next = logits_next.argmax(dim=-1)
                    label_idx_next = lm_input['label_idx_til']
                else:
                    raise NotImplementedError(f"Unknown IL mode {params.il_mode}")

                preds_next, label_idx_next = accelerator.gather_for_metrics((preds_next, label_idx_next))
                label_idx_next = label_idx_next.detach().cpu().numpy()
                acc_list_next.extend([1 if p == t else 0 for p, t in zip(preds_next, label_idx_next)])

        acc_next_task = np.round(np.mean(acc_list_next) * 100, 3)
    else:
        acc_next_task = None

    print(f"acc: {acc}")
    print(f"acc_next_task: {acc_next_task}")

    model.train()
    return acc, acc_next_task

def evaluate_sent_level_acc_with_classifier_adapter(model, classifier_list, cur_task_id, eval_data_loader, tokenizer, accelerator, params, idx2label):
    '''
        Evaluate the accuracy with classifiers and adapters for sentence-level classification tasks 

        Return:
         - Acc (%)
    '''
    model.eval()
    acc_list_all = []

    def set_adapter_to_task(t_id):
        try:
            model.set_adapter('task-%d'%(t_id))
        except:
            try:
                model.set_active_adapters('task-%d'%(t_id))
            except:
                raise NotImplementedError()

            
    with torch.no_grad():
        for lm_input in eval_data_loader: 
    
            if params.il_mode == 'CIL':
                # NOTE: requires O(num_tasks) forward!
                logits_list = []
                for t_id in range(cur_task_id+1):
                    set_adapter_to_task(t_id)
                    extracted_features = obtain_features(params=params, 
                                                        model=model, 
                                                        lm_input=lm_input,
                                                        tokenizer=tokenizer)
                    # classifier를 extracted_features와 같은 dtype으로 맞춤
                    if hasattr(classifier_list[t_id], 'weight') and classifier_list[t_id].weight.dtype != extracted_features.dtype:
                        classifier_list[t_id] = classifier_list[t_id].to(extracted_features.dtype)
                    logits_list.append(classifier_list[t_id](extracted_features))
                logits = torch.cat(logits_list,dim=-1)
                preds = logits.argmax(dim=-1)
                label_idx = lm_input['label_idx_cil']

            elif params.il_mode == 'TIL':
                # NOTE: In the TIL mode, the cur_task_id should be set as eval_task_id!
                set_adapter_to_task(cur_task_id)
                extracted_features = obtain_features(params=params, 
                                                    model=model, 
                                                    lm_input=lm_input,
                                                    tokenizer=tokenizer)
                # classifier를 extracted_features와 같은 dtype으로 맞춤
                if hasattr(classifier_list[cur_task_id], 'weight') and classifier_list[cur_task_id].weight.dtype != extracted_features.dtype:
                    classifier_list[cur_task_id] = classifier_list[cur_task_id].to(extracted_features.dtype)
                logits = classifier_list[cur_task_id](extracted_features) 
                preds = logits.argmax(dim=-1)
                label_idx = lm_input['label_idx_til']
            else:
                raise NotImplementedError()
            # Gather from different processes in DDP
            preds, label_idx = accelerator.gather_for_metrics((preds, label_idx))
            label_idx = label_idx.detach().cpu().numpy()
            acc_list = [1 if pred==target else 0 for pred, target in zip(preds,label_idx)]
            acc_list_all.extend(acc_list)
            acc = np.round(np.mean(acc_list_all)*100,3)
    
    model.train()

    return acc

def evaluate_sent_level_acc_with_classifier_model_list(model_list, classifier_list, cur_task_id, eval_data_loader, tokenizer, accelerator, params):
    '''
        Evaluate the accuracy with classifier and model_list for sentence-level classification tasks

        Return:
         - Acc (%)
    '''
    assert params.il_mode == 'CIL', 'NotImplemented for il mode %s'%(params.il_mode)
    model_list[cur_task_id].eval()
    acc_list_all = []
            
    with torch.no_grad():
        for lm_input in eval_data_loader: 
            logits = []
            for t_id in range(cur_task_id+1):
                # For Distributed Data Parallel
                if hasattr(model_list[t_id],'module'):
                    model = model_list[t_id].module
                else:
                    model = model_list[t_id]
                extracted_feature = obtain_features(params=params, 
                                                    model=model, 
                                                    lm_input=lm_input, 
                                                    tokenizer=tokenizer)
                # classifier를 extracted_feature와 같은 dtype으로 맞춤
                if hasattr(classifier_list[t_id], 'weight') and classifier_list[t_id].weight.dtype != extracted_feature.dtype:
                    classifier_list[t_id] = classifier_list[t_id].to(extracted_feature.dtype)
                logits.append(classifier_list[t_id](extracted_feature)) 

            logits = torch.concatenate(logits,dim=-1)
            preds = logits.argmax(dim=-1)
            label_idx = lm_input['label_idx_cil']
            # Gather from different processes in DDP
            preds, label_idx = accelerator.gather_for_metrics((preds, label_idx))

            acc_list = [1 if pred==target else 0 for pred, target in zip(preds,label_idx)]
            acc_list_all.extend(acc_list)
    
    acc = np.round(np.mean(acc_list_all)*100,3)
    model_list[cur_task_id].train()
    
    model.train()

    return acc

def evaluate_with_generation_ICL(model, train_data_loader, eval_data_loader, tokenizer, accelerator, params):
    '''
        Evaluate the accuracy using generation for sentence-level classification tasks

        Return:
         - Acc (%)
    '''

    assert params.il_mode == 'IIL'

    model.eval()
    acc_list_all = []
    save_dict = {}

    train_data_loader = DataLoader(train_data_loader.dataset, 
                                    batch_size=params.batch_size, 
                                    shuffle=False if params.ICL_same_instance or params.ICL_same_concept else True,
                                    drop_last=False)
    train_data_loader = accelerator.prepare(train_data_loader)
    total_sample = len(eval_data_loader.dataset)

    with torch.no_grad():
        for train_lm_input, eval_lm_input in zip(train_data_loader, eval_data_loader): 
            bs = eval_lm_input['input_ids'].shape[0]
            prompted_input_all = []

            # Prepare eval sample
            eval_inputs = tokenizer.batch_decode(eval_lm_input['input_ids'],skip_special_tokens=True) 

            for bs_i in range(bs):
                # Prepare N-shot sample                
                train_input_ids = []
                train_target = []
                if params.ICL_same_concept:
                    eval_concept = eval_lm_input['concept_id'][bs_i]
                    match_idx = torch.where(train_data_loader.dataset['concept_id']==eval_concept.item())[0]
                    select_idx = np.random.randint(match_idx.shape[0],size=params.ICL_n_shot)
                    for s_idx in select_idx:
                        s_idx = int(s_idx)
                        train_input_ids.append(train_data_loader.dataset['input_ids'][s_idx])
                        train_target.append(train_data_loader.dataset['target'][s_idx])
                elif params.ICL_same_instance:
                    if params.ICL_n_shot>1:
                        select_idx = np.random.randint(total_sample,size=params.ICL_n_shot-1)
                        for s_idx in select_idx:
                            s_idx = int(s_idx)
                            train_input_ids.append(train_data_loader.dataset['input_ids'][s_idx])
                            train_target.append(train_data_loader.dataset['target'][s_idx])
                    train_input_ids.append(train_lm_input['input_ids'][bs_i])
                    train_target.append(train_lm_input['target'][bs_i])
                else:
                    select_idx = np.random.randint(total_sample,size=params.ICL_n_shot)
                    for s_idx in select_idx:
                        s_idx = int(s_idx)
                        train_input_ids.append(train_data_loader.dataset['input_ids'][s_idx])
                        train_target.append(train_data_loader.dataset['target'][s_idx])

                train_inputs = tokenizer.batch_decode(train_input_ids,skip_special_tokens=True)
                shuffle_demonstration = list(range(params.ICL_n_shot))
                shuffle(shuffle_demonstration)
                _train_input_list = []
                for i_shot in shuffle_demonstration:
                    _train_input_list.append(f'{train_inputs[i_shot]} {train_target[i_shot]}')

                prompted_input = get_prompt_ICL(_train_input_list, 
                                                eval_inputs[bs_i],
                                                )
                prompted_input_all.append(prompted_input)

            prompted_eval_lm_input = tokenizer(prompted_input_all,
                                                padding='longest',
                                                return_tensors="pt").data
            prompted_eval_lm_input = {
                'input_ids': prompted_eval_lm_input['input_ids'].to(model.device),
                'attention_mask': prompted_eval_lm_input['attention_mask'].to(model.device)
            }

            generate_ids = obtain_generate_ids(params=params, 
                                                model=model, 
                                                lm_input=prompted_eval_lm_input, 
                                                tokenizer=tokenizer)

            # Gather from different processes in DDP
            generate_ids = accelerator.pad_across_processes(
                                                generate_ids, 
                                                dim=1, 
                                                pad_index=tokenizer.eos_token_id, 
                                                pad_first=False)
            generate_ids = accelerator.gather_for_metrics(generate_ids)

            generate_seq = tokenizer.batch_decode(generate_ids,skip_special_tokens=True)
            generate_seq = [pred.strip(' \n').lower().split(tokenizer.eos_token)[0].split('\n')[0].strip() 
                            for pred in generate_seq]

            # Compare the target directly
            target_output = eval_lm_input['target']

            acc_list = [1 if pred.strip().lower()==target.strip().lower() else 0 for pred, target in zip(generate_seq,target_output)]
            acc_list_all.extend(acc_list)

            if params.il_mode == 'IIL':
                for i, (pred, target) in enumerate(zip(generate_seq,target_output)):
                    save_dict[eval_lm_input['instance_id'][i].item()] = {
                        'instance_id': eval_lm_input['instance_id'][i].item(),
                        'relation_id': eval_lm_input['relation_id'][i].item(),
                        'concept_id': eval_lm_input['concept_id'][i].item(),
                        'prediton': pred.strip().lower(),
                        'target': target.strip().lower(),
                    }

            if len(acc_list_all)%(5*bs)==0: 
                logger.info('ICL Progress %.2f%%(=%d/%d)'%
                    (len(acc_list_all)/total_sample*100,
                    len(acc_list_all),
                    total_sample)
                )
    
    acc = np.round(np.mean(acc_list_all)*100,3)
    model.train()

    if params.il_mode == 'IIL':
        return acc, save_dict

    return acc

# ------------------------------------- sentence level evaluation ----------------------------------------------
def evaluate_word_level_acc_with_classifier(model, classifier_list, cur_task_id, eval_data_loader, next_eval_data_loader, tokenizer, accelerator, params, idx2label):
    '''
        Evaluate the accuracy with classifier for word-level classification tasks

        Return:
         - macro f1 (%)
         - macro f1 for next task (if applicable)
    '''
    assert params.il_mode == 'CIL', 'NotImplemented for il mode %s' % (params.il_mode)
    model.eval()

    with torch.no_grad():
        # Evaluate current task (cur_task_id)
        gold_lines = []
        pred_lines = []

        for lm_input in eval_data_loader:
            extracted_features = obtain_features(params=params,
                                                 model=model,
                                                 lm_input=lm_input,
                                                 tokenizer=tokenizer)

            logits = torch.cat([classifier(extracted_features) for classifier in classifier_list[:cur_task_id + 1]], dim=-1)
            preds = logits.argmax(dim=-1)
            label_idx = lm_input['label_idx_cil']
            # Gather from different processes in DDP
            preds, label_idx = accelerator.gather_for_metrics((preds, label_idx))
            preds, label_idx = preds.cpu(), label_idx.cpu()

            for _sent_gold, _sent_pred in zip(label_idx, preds):
                gold_lines.append([idx2label[_gold.item()] for _gold, _pred in zip(_sent_gold, _sent_pred) if _gold.item() != -100])
                pred_lines.append([idx2label[_pred.item()] for _gold, _pred in zip(_sent_gold, _sent_pred) if _gold.item() != -100])

        # Micro F1 (average over all instances)
        mi_f1 = f1_score(gold_lines, pred_lines) * 100
        # Macro F1 (average over each class)
        ma_f1 = f1_score(gold_lines, pred_lines, average='macro') * 100

        logger.info('Evaluate current task: ma_f1 = %.2f, mi_f1 = %.2f' % (ma_f1, mi_f1))

    # Evaluate next task (cur_task_id + 1), if it exists
    if next_eval_data_loader is not None and cur_task_id + 1 < len(classifier_list):
        with torch.no_grad():
            gold_lines_next = []
            pred_lines_next = []

            for lm_input in next_eval_data_loader:
                extracted_features = obtain_features(params=params,
                                                     model=model,
                                                     lm_input=lm_input,
                                                     tokenizer=tokenizer)

                logits_next = torch.cat([classifier(extracted_features) for classifier in classifier_list[:cur_task_id + 2]], dim=-1)
                preds_next = logits_next.argmax(dim=-1)
                label_idx_next = lm_input['label_idx_cil']
                # Gather from different processes in DDP
                preds_next, label_idx_next = accelerator.gather_for_metrics((preds_next, label_idx_next))
                preds_next, label_idx_next = preds_next.cpu(), label_idx_next.cpu()

                for _sent_gold, _sent_pred in zip(label_idx_next, preds_next):
                    gold_lines_next.append([idx2label[_gold.item()] for _gold, _pred in zip(_sent_gold, _sent_pred) if _gold.item() != -100])
                    pred_lines_next.append([idx2label[_pred.item()] for _gold, _pred in zip(_sent_gold, _sent_pred) if _gold.item() != -100])

            # Micro F1 for next task
            mi_f1_next = f1_score(gold_lines_next, pred_lines_next) * 100
            # Macro F1 for next task
            ma_f1_next = f1_score(gold_lines_next, pred_lines_next, average='macro') * 100

            logger.info('Evaluate next task: ma_f1 = %.2f, mi_f1 = %.2f' % (ma_f1_next, mi_f1_next))
    else:
        ma_f1_next = None  # No next task available for evaluation

    model.train()

    return ma_f1, ma_f1_next

def evaluate_word_level_acc_with_classifier_adapter(model, classifier_list, cur_task_id, eval_data_loader, tokenizer, accelerator, params, idx2label):
    '''
        Evaluate the accuracy with classifiers and adapters for word-level classification tasks

        Return:
         - macro f1 (%)
    '''
    assert params.il_mode == 'CIL', 'NotImplemented for il mode %s'%(params.il_mode)
    model.eval()

    def set_adapter_to_task(t_id):
        try:
            model.set_adapter('task-%d'%(t_id))
        except:
            try:
                model.set_active_adapters('task-%d'%(t_id))
            except:
                raise NotImplementedError()
            
    with torch.no_grad():
        # Only test once because the last test set contains all labels
        gold_lines = []
        pred_lines = []

        for lm_input in eval_data_loader: 
            logits_list = []
            for t_id in range(cur_task_id+1):
                set_adapter_to_task(t_id)
                extracted_features = obtain_features(params=params, 
                                                    model=model, 
                                                    lm_input=lm_input,
                                                    tokenizer=tokenizer)
                # classifier를 extracted_features와 같은 dtype으로 맞춤
                if hasattr(classifier_list[t_id], 'weight') and classifier_list[t_id].weight.dtype != extracted_features.dtype:
                    classifier_list[t_id] = classifier_list[t_id].to(extracted_features.dtype)
                # We use the O logits in the first classifier
                logits_list.append(classifier_list[t_id](extracted_features))
            logits = torch.cat(logits_list,dim=-1)

            # Remove the prompt token manually
            if params.PEFT_type == 'PromptTuning':
                logits = logits[:,params.PEFT_num_virtual_tokens:,:] 

            preds = logits.argmax(dim=-1)
            label_idx = lm_input['label_idx_cil']
            # Gather from different processes in DDP
            preds, label_idx = accelerator.gather_for_metrics((preds, label_idx))
            preds, label_idx = preds.cpu(), label_idx.cpu()

            for _sent_gold, _sent_pred in zip(label_idx,preds):
                gold_lines.append([idx2label[_gold.item()] for _gold, _pred in zip(_sent_gold,_sent_pred) if _gold.item()!=-100])
                pred_lines.append([idx2label[_pred.item()] for _gold, _pred in zip(_sent_gold,_sent_pred) if _gold.item()!=-100])

        # micro f1 (average over all instances)
        mi_f1 = f1_score(gold_lines, pred_lines)*100
        # macro f1 (default, average over each class)
        ma_f1 = f1_score(gold_lines, pred_lines, average='macro')*100

        logger.info('Evaluate: ma_f1 = %.2f, mi_f1=%.2f'%(ma_f1,mi_f1))
    
    model.train()

    return ma_f1

# ==============================================================================================
# New utility: evaluate_sent_level_acc_with_lmhead
#   • 백본은 고정, **LM Head** 의 vocab-logit으로 “지금까지 학습한 모든 클래스”를 평가
#   • current task 와 next task(존재할 경우) 를 각각 accuracy 로 반환
# ==============================================================================================
def _gold_token_ids(lm_input, tokenizer, device, idx2label):
    """
    ‣ 우선순위
      ① lm_input["target"]               ─ training-like batch
      ② lm_input["labels_with_ans"]      ─ training batch(텍스트 없음)
      ③ lm_input["label_idx_cil"]        ─ dev/test 배치
    """
    import logging
    logger = logging.getLogger()
    
    # ① target 문자열
    if "target" in lm_input:
        enc = tokenizer(
            lm_input["target"], add_special_tokens=False, padding=False
        )
        ids = [seq[0] for seq in enc["input_ids"]]
        logger.debug(f"[Gold Token] Using 'target': {lm_input['target'][:3]} → {ids[:3]}")
        return torch.tensor(ids, device=device)

    # ② labels_with_ans 텐서
    if "labels_with_ans" in lm_input:
        lbls = lm_input["labels_with_ans"]
        first_ids = []
        for row in lbls:
            non_pad = (row != -100).nonzero(as_tuple=False)
            first_ids.append(
                row[non_pad[0]].item() if non_pad.numel()
                else tokenizer.eos_token_id
            )
        logger.debug(f"[Gold Token] Using 'labels_with_ans': first_ids[:3] = {first_ids[:3]}")
        return torch.tensor(first_ids, device=device)

    # ③ label_idx_cil → idx2label 매핑
    if "label_idx_cil" in lm_input:
        idx = lm_input["label_idx_cil"].tolist()
        text = [idx2label[i] for i in idx]
        enc = tokenizer(text, add_special_tokens=False, padding=False)
        ids = [seq[0] for seq in enc["input_ids"]]
        logger.debug(f"[Gold Token] Using 'label_idx_cil': idx[:3]={idx[:3]} → text[:3]={text[:3]} → ids[:3]={ids[:3]}")
        return torch.tensor(ids, device=device)

    # 안전장치
    B = lm_input["input_ids"].size(0)
    logger.warning(f"[Gold Token] Fallback to eos_token_id for batch size {B}")
    return torch.full((B,), tokenizer.eos_token_id, device=device)


def evaluate_sent_level_acc_with_lmhead(
        model, cur_task_id,
        eval_data_loader, next_eval_data_loader,
        tokenizer, accelerator, params, idx2label):

    import logging
    logger = logging.getLogger()
    
    model.eval()
    lm_head = model.get_output_embeddings()

    def _acc(loader):
        if loader is None:
            return None
        correct = total = 0
        with torch.no_grad():
            for batch_idx, lm_input in enumerate(loader):
                feats = obtain_features(params, model, lm_input, tokenizer)
                
                # dtype 일관성 확보
                if feats.dtype != lm_head.weight.dtype:
                    feats = feats.to(lm_head.weight.dtype)
                
                logits = lm_head(feats)
                preds = logits.argmax(dim=-1)
                gold  = _gold_token_ids(lm_input, tokenizer, preds.device, idx2label)

                # 디버깅 정보 출력 (첫 번째 배치만)
                if batch_idx == 0 and accelerator.is_main_process:
                    logger.info(f"[LM-Head Eval Debug] Batch 0:")
                    logger.info(f"  Feature shape: {feats.shape}, dtype: {feats.dtype}")
                    logger.info(f"  LM-Head weight shape: {lm_head.weight.shape}, dtype: {lm_head.weight.dtype}")
                    logger.info(f"  Logits shape: {logits.shape}, vocab_size: {logits.shape[-1]}")
                    logger.info(f"  Predictions (first 5): {preds[:5].tolist()}")
                    logger.info(f"  Gold tokens (first 5): {gold[:5].tolist()}")
                    
                    # 토큰 ID → 텍스트 변환 확인
                    if len(preds) > 0:
                        pred_texts = [tokenizer.decode([pid.item()]) for pid in preds[:5]]
                        gold_texts = [tokenizer.decode([gid.item()]) for gid in gold[:5]]
                        logger.info(f"  Pred texts: {pred_texts}")
                        logger.info(f"  Gold texts: {gold_texts}")

                preds, gold = accelerator.gather_for_metrics((preds, gold))
                batch_correct = (preds == gold).sum().item()
                batch_total = gold.numel()
                correct += batch_correct
                total += batch_total
                
                # 첫 번째 배치 정확도 로그
                if batch_idx == 0 and accelerator.is_main_process:
                    batch_acc = 100 * batch_correct / max(batch_total, 1)
                    logger.info(f"  Batch 0 accuracy: {batch_acc:.2f}% ({batch_correct}/{batch_total})")
                    
        return np.round(100 * correct / max(total, 1), 3)

    acc_cur  = _acc(eval_data_loader)
    acc_next = _acc(next_eval_data_loader)
    
    if accelerator.is_main_process:
        logger.info(f"[LM-Head Eval] Current task accuracy: {acc_cur}%")
        if acc_next is not None:
            logger.info(f"[LM-Head Eval] Next task accuracy: {acc_next}%")
    
    model.train()
    return acc_cur, acc_next

# ==================================================================================================================

# ====================== the evaluation metric computed after whole IL training ==========================================================================
def compute_average_acc(result_matrix):
    '''
        Compute the average accuracy (after training the last task) for continual learning

        Params:
            - result_matrix: (NUM_TASK,NUM_TASK); the result for each task in every CL step

        Returns:
            - aver_acc: float
    '''
    return np.mean(result_matrix[-1,:])

def compute_average_inc_acc(result_matrix):
    '''
        Compute the average incremental accuracy (average over all incremental steps)

        Params:
            - result_matrix: (NUM_TASK,NUM_TASK); the result for each task in every CL step

        Returns:
            - aver_inc_acc: float
    '''
    NUM_TASK = result_matrix.shape[0]
    # Compute the average acc of each incremental steps
    aver_acc_list = [np.mean(result_matrix[i,:i+1]) for i in range(NUM_TASK)]
    if len(aver_acc_list)==0:
        return -1
    return np.mean(aver_acc_list)

def compute_forgetting(result_matrix):
    '''
        Compute the forgetting for continual learning

        Params:
            - result_matrix: (NUM_TASK,NUM_TASK); the result for each task in every CL step

        Returns:
            - fgt: float
    '''
    NUM_TASK = result_matrix.shape[0]
    fgt_list = []
    for t_id in range(0,NUM_TASK-1):
        fgt_list.append(np.max(result_matrix[:,t_id])-result_matrix[-1,t_id])

    if len(fgt_list)==0:
        return -1
    return np.mean(fgt_list)

def compute_backward_transfer(result_matrix):
    '''
        Compute the backward transfer for continual learning

        Params:
            - result_matrix: (NUM_TASK,NUM_TASK); the result for each task in every CL step

        Returns:
            - bwt: float
    '''
    NUM_TASK = result_matrix.shape[0]
    bwt_list = []
    for t_id in range(0,NUM_TASK-1):
        bwt_list.append(result_matrix[-1,t_id]-result_matrix[t_id,t_id])

    if len(bwt_list)==0:
        return -1
    return np.mean(bwt_list)

def compute_forward_transfer(result_matrix):
    '''
        Compute the forward transfer for continual learning

        Params:
            - result_matrix: (NUM_TASK, NUM_TASK); the result for each task in every CL step

        Returns:
            - fwt: float
    '''
    NUM_TASK = result_matrix.shape[0]
    fwt_list = []

    # Forward Transfer 계산
    for t_id in range(1,NUM_TASK):
        # 이전 태스크(t_id-1)에 대한 성능과 현재 태스크(t_id)에 대한 초기 성능 확인
        R_k_minus_1_k = result_matrix[t_id, t_id]
        b_k = result_matrix[t_id - 1, t_id]
        #print(f"R_k_minus_1_k: {R_k_minus_1_k}")
        #print(f"b_k: {b_k}")


        fwt_list.append(R_k_minus_1_k - b_k)

    if len(fwt_list) == 0:
        return -1

    return np.mean(fwt_list)

# ==================================================================================================================
