import logging
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from seqeval.metrics import f1_score
from random import shuffle
from copy import deepcopy

from utils.wrapmodel import WrapModel
from utils.backbone import obtain_features
from utils.classifier import CosineLinear

logger = logging.getLogger()

def probing_on_all_task(params, task_id, CL_dataset, train_loader_list, test_loader_list, model, tokenizer, accelerator) -> dict:
    """
        Probing: It re-trains classifiers on data of ALL tasks (CIL mode) or up to the next task (TIL mode).
        The probing performance represents what PTMs really know about the IL tasks.

        Return:
            - prob_result_dict: {
                'LinearProb': [태스크별 정확도 리스트],  # Linear Probing 성능
                # 다른 분류기 결과도 포함
            }
    """
    il_mode = params.il_mode
    assert il_mode in ['CIL', 'TIL'], 'NotImplemented for il_mode %s' % (il_mode)

    total_num_task = CL_dataset.continual_config['NUM_TASK']
    cur_num_class_list = CL_dataset.continual_config['CUR_NUM_CLASS']

    if il_mode == 'CIL':
        # CIL 모드에서는 전체 태스크의 데이터를 사용
        num_task = len(train_loader_list)  # train_loader_list의 실제 길이 (전체 태스크)
        cur_num_class = cur_num_class_list  # 모든 태스크의 클래스 수 리스트
        num_class = sum(cur_num_class)
    else:
        # TIL 모드에서는 현재 태스크와 다음 태스크까지의 데이터를 사용
        num_task = len(train_loader_list)  # 전달된 train_loader_list의 길이
        cur_num_class = cur_num_class_list[:num_task]
        num_class = cur_num_class  # 태스크별 클래스 수 리스트

    # 모델 준비
    if hasattr(model, 'module'):
        model = model.module
    if isinstance(model, WrapModel):
        model = model.model
    if hasattr(model, 'backbone_model'):
        model = model.backbone_model

    # Step 1: 모든 특징 추출
    with torch.no_grad():
        train_feature_list = []
        train_label_idx_list = []
        for t_id in range(num_task):
            _train_feature_list = []
            _train_label_idx_list = []
            for lm_input in train_loader_list[t_id]:
                extracted_features = obtain_features(params=params,
                                                     model=model,
                                                     lm_input=lm_input,
                                                     tokenizer=tokenizer)
                if il_mode == 'CIL':
                    label_idx = lm_input['label_idx_cil']
                else:
                    label_idx = lm_input['label_idx_til']

                extracted_features, label_idx = accelerator.gather_for_metrics((extracted_features, label_idx))
                _train_feature_list.append(extracted_features.detach().cpu())
                _train_label_idx_list.append(label_idx.cpu())

            _train_feature_list = torch.cat(_train_feature_list, dim=0)
            _train_label_idx_list = torch.cat(_train_label_idx_list, dim=0)

            train_feature_list.append(_train_feature_list)
            train_label_idx_list.append(_train_label_idx_list)

        if il_mode == 'CIL':
            train_features_all = torch.cat(train_feature_list, dim=0)
            train_label_idx_all = torch.cat(train_label_idx_list, dim=0)
        else:
            train_features_all = train_feature_list
            train_label_idx_all = train_label_idx_list

        del train_feature_list, train_label_idx_list
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        # Step 2: 분류기 학습
        if il_mode == 'CIL':
            train_features = train_features_all
            train_label_idx = train_label_idx_all

            feature_dim = train_features.shape[-1]

            # 분류기 초기화 (출력 차원은 전체 클래스 수)
            linear_layer = nn.Linear(in_features=feature_dim, out_features=num_class, bias=False).cuda().train()
            coslinear_layer = CosineLinear(in_features=feature_dim, out_features=num_class).cuda().train()

            prototype_layer = nn.Linear(in_features=feature_dim, out_features=num_class, bias=False).cuda()
            cosprototype_layer = CosineLinear(in_features=feature_dim, out_features=num_class).cuda()

            opt_linear = Adam(linear_layer.parameters(), lr=0.001)
            opt_coslinear = Adam(coslinear_layer.parameters(), lr=0.001)
            loss_fct = nn.CrossEntropyLoss()
            num_chunk = max(train_features.shape[0] // 128, 1)
            loss_list_epochs_linear = []
            loss_list_epochs_coslinear = []

            # 프로토타입 계산을 위한 변수 초기화
            cur_class_prototypes = torch.zeros_like(prototype_layer.weight.data).cuda()
            cnt_class_samples = {class_idx: 0 for class_idx in range(num_class)}

            for e_id in range(20):

                loss_list_linear = []
                loss_list_coslinear = []
                shuffle_idx = torch.randperm(train_features.shape[0])

                for selected_idx in torch.chunk(shuffle_idx, num_chunk):
                    _feature, _label_idx = train_features[selected_idx], train_label_idx[selected_idx]
                    _feature, _label_idx = _feature.cuda(), _label_idx.cuda()

                    if params.classification_type == 'word-level':
                        label_mask = (_label_idx != -100)
                        _feature = _feature[label_mask]  # flatten
                        _label_idx = _label_idx[label_mask]  # flatten

                    # Train Linear Classifier
                    logits_linear = linear_layer(_feature)
                    loss_linear = loss_fct(logits_linear, _label_idx)
                    opt_linear.zero_grad()
                    loss_linear.backward()
                    opt_linear.step()

                    # Train Cosine Linear Classifier
                    logits_coslinear = coslinear_layer(_feature)
                    loss_coslinear = loss_fct(logits_coslinear, _label_idx)
                    opt_coslinear.zero_grad()
                    loss_coslinear.backward()
                    opt_coslinear.step()

                    loss_list_linear.append(loss_linear.item())
                    loss_list_coslinear.append(loss_coslinear.item())

                    # Compute Class Center for Prototype/CosinePrototype Classifier (Only need one epoch)
                    if e_id == 0:
                        for class_idx in torch.unique(_label_idx):
                            class_mask = (_label_idx == class_idx)
                            cnt_class_samples[class_idx.item()] += class_mask.sum().item()
                            cur_class_prototypes[class_idx.item()] += _feature[class_mask].detach().sum(dim=0)

                loss_list_epochs_linear.append(np.mean(loss_list_linear))
                loss_list_epochs_coslinear.append(np.mean(loss_list_coslinear))

            # Set the weight of the Prototype/CosinePrototype Classifier
            for class_idx in cnt_class_samples.keys():
                if cnt_class_samples[class_idx] == 0:
                    continue
                prototype_layer.weight.data[class_idx] = cur_class_prototypes[class_idx] / cnt_class_samples[class_idx]
                cosprototype_layer.weight.data[class_idx] = cur_class_prototypes[class_idx] / cnt_class_samples[class_idx]

            del train_features, train_label_idx
            torch.cuda.empty_cache()

        else:
            # TIL 모드
            feature_dim = train_features_all[0].shape[-1]

            linear_layer = nn.ModuleList([
                nn.Linear(in_features=feature_dim, out_features=cur_num_class[t_id], bias=False).cuda().train()
                for t_id in range(num_task)
            ])
            coslinear_layer = nn.ModuleList([
                CosineLinear(in_features=feature_dim, out_features=cur_num_class[t_id]).cuda().train()
                for t_id in range(num_task)
            ])

            prototype_layer = nn.ModuleList([
                nn.Linear(in_features=feature_dim, out_features=cur_num_class[t_id], bias=False).cuda()
                for t_id in range(num_task)
            ])
            cosprototype_layer = nn.ModuleList([
                CosineLinear(in_features=feature_dim, out_features=cur_num_class[t_id]).cuda()
                for t_id in range(num_task)
            ])

            opt_linear_list = [Adam(clf.parameters(), lr=0.001) for clf in linear_layer]
            opt_coslinear_list = [Adam(clf.parameters(), lr=0.001) for clf in coslinear_layer]
            loss_fct = nn.CrossEntropyLoss()

            # 프로토타입 계산을 위한 변수 초기화
            cur_class_prototypes = [
                torch.zeros_like(prototype_layer[t_id].weight.data).cuda()
                for t_id in range(num_task)
            ]
            cnt_class_samples = [
                {class_idx: 0 for class_idx in range(cur_num_class[t_id])}
                for t_id in range(num_task)
            ]

            for t_id in range(num_task):
                train_features = train_features_all[t_id]
                train_label_idx = train_label_idx_all[t_id]

                num_chunk = max(train_features.shape[0] // 128, 1)
                loss_list_epochs_linear = []
                loss_list_epochs_coslinear = []

                for e_id in range(20):

                    loss_list_linear = []
                    loss_list_coslinear = []
                    shuffle_idx = torch.randperm(train_features.shape[0])

                    for selected_idx in torch.chunk(shuffle_idx, num_chunk):
                        _feature, _label_idx = train_features[selected_idx], train_label_idx[selected_idx]
                        _feature, _label_idx = _feature.cuda(), _label_idx.cuda()

                        if params.classification_type == 'word-level':
                            label_mask = (_label_idx != -100)
                            _feature = _feature[label_mask]  # flatten
                            _label_idx = _label_idx[label_mask]  # flatten

                        # Train Linear Classifier
                        logits_linear = linear_layer[t_id](_feature)
                        loss_linear = loss_fct(logits_linear, _label_idx)
                        opt_linear_list[t_id].zero_grad()
                        loss_linear.backward()
                        opt_linear_list[t_id].step()

                        # Train Cosine Linear Classifier
                        logits_coslinear = coslinear_layer[t_id](_feature)
                        loss_coslinear = loss_fct(logits_coslinear, _label_idx)
                        opt_coslinear_list[t_id].zero_grad()
                        loss_coslinear.backward()
                        opt_coslinear_list[t_id].step()

                        loss_list_linear.append(loss_linear.item())
                        loss_list_coslinear.append(loss_coslinear.item())

                        # Compute Class Center for Prototype/CosinePrototype Classifier (Only need one epoch)
                        if e_id == 0:
                            for class_idx in torch.unique(_label_idx):
                                class_mask = (_label_idx == class_idx)
                                cnt_class_samples[t_id][class_idx.item()] += class_mask.sum().item()
                                cur_class_prototypes[t_id][class_idx.item()] += _feature[class_mask].detach().sum(dim=0)

                    loss_list_epochs_linear.append(np.mean(loss_list_linear))
                    loss_list_epochs_coslinear.append(np.mean(loss_list_coslinear))

                # Set the weight of the Prototype/CosinePrototype Classifier
                for class_idx in cnt_class_samples[t_id].keys():
                    if cnt_class_samples[t_id][class_idx] == 0:
                        continue
                    prototype_layer[t_id].weight.data[class_idx] = cur_class_prototypes[t_id][class_idx] / cnt_class_samples[t_id][class_idx]
                    cosprototype_layer[t_id].weight.data[class_idx] = cur_class_prototypes[t_id][class_idx] / cnt_class_samples[t_id][class_idx]

                del train_features, train_label_idx
                torch.cuda.empty_cache()

            del train_features_all, train_label_idx_all

        # Step 3: 분류기 평가
        with torch.no_grad():
            prob_result_dict = {}
            for metric_name, _classifier in zip(['LinearProb', 'CosineLinearProb', 'PrototypeProb', 'CosinePrototypeProb'],
                                                [linear_layer, coslinear_layer, prototype_layer, cosprototype_layer]):
                if il_mode == 'CIL':
                    _classifier.eval()
                else:
                    for clf in _classifier:
                        clf.eval()

                if params.classification_type == 'sentence-level':
                    tasks_acc_list = []
                    for t_id in range(total_num_task):
                        if il_mode == 'TIL' and t_id >= num_task:
                            tasks_acc_list.append(-1)  # TIL 모드에서 포함되지 않은 태스크는 -1로 표시
                            continue
                        acc_list = []
                        for lm_input in test_loader_list[t_id]:
                            _feature = obtain_features(params=params,
                                                       model=model,
                                                       lm_input=lm_input,
                                                       tokenizer=tokenizer)
                            if il_mode == 'CIL':
                                _label_idx = lm_input['label_idx_cil']
                                _feature, _label_idx = accelerator.gather_for_metrics((_feature, _label_idx))
                                logits = _classifier(_feature)
                            else:
                                _label_idx = lm_input['label_idx_til']
                                _feature, _label_idx = accelerator.gather_for_metrics((_feature, _label_idx))
                                logits = _classifier[t_id](_feature)
                            preds = logits.argmax(dim=-1).detach().cpu()
                            #print(f"preds: {preds}")
                            _label_idx = _label_idx.cpu()
                            #print(f"_label_idx: {_label_idx}")
                            acc_list.extend([(_pred == _label).item() for _pred, _label in zip(preds, _label_idx)])
                        if len(acc_list) > 0:
                            task_accuracy = np.round(np.mean(acc_list) * 100, 3)
                        else:
                            task_accuracy = -1  # 데이터가 없는 경우 -1로 표시
                        tasks_acc_list.append(task_accuracy)
                        #print(f"metric_name: {metric_name}")
                        #print(f"tasks_acc_list: {tasks_acc_list}")
                        torch.cuda.empty_cache()
                else:
                    # 필요 시 word-level 평가 구현
                    raise NotImplementedError('Word-level classification not implemented for this scenario')

                prob_result_dict[metric_name] = deepcopy(tasks_acc_list)

        # 필요 시 분류기 저장
        if params.save_probing_classifiers:
            if il_mode == 'CIL':
                torch.save(
                    {
                        'linear_layer': linear_layer.cpu(),
                        'coslinear_layer': coslinear_layer.cpu(),
                        'prototype_layer': prototype_layer.cpu(),
                        'cosprototype_layer': cosprototype_layer.cpu(),
                    },
                    os.path.join(params.dump_path, 'probing_classifiers_task%d.pth' % (task_id))
                )
            else:
                torch.save(
                    {
                        'linear_layer': [layer.cpu() for layer in linear_layer],
                        'coslinear_layer': [layer.cpu() for layer in coslinear_layer],
                        'prototype_layer': [layer.cpu() for layer in prototype_layer],
                        'cosprototype_layer': [layer.cpu() for layer in cosprototype_layer],
                    },
                    os.path.join(params.dump_path, 'probing_classifiers_task%d.pth' % (task_id))
                )

    else:
        # Non-main processes return placeholder results
        num_tasks_for_results = total_num_task
        prob_result_dict = {
            'LinearProb': [-1] * num_tasks_for_results,
            'CosineLinearProb': [-1] * num_tasks_for_results,
            'PrototypeProb': [-1] * num_tasks_for_results,
            'CosinePrototypeProb': [-1] * num_tasks_for_results,
        }

    return prob_result_dict




# def probing_on_all_task(params, task_id, CL_dataset, train_loader_list, test_loader_list, model, tokenizer, accelerator) -> dict:
#     """
#         Probing: It re-trains Linear/CosineLinear/Prototype/CosinePrototype classifiers respecitvely
#         on data of ALL tasks.
#         The probing performance represents that what PTMs really know about the IL tasks.

#         Return:
#             - prob_result_dict:{
            
#                 'LinearProb':[95.2, 94.7, ..., 91.3],          # Linear Probing Performance

#                 'CosineLinearProb':[95.2, 94.7, ..., 91.3],    # Cosine Linear Probing Performance

#                 'PrototypeProb':[95.2, 94.7, ..., 91.3],       # Prototype Probing Performance

#                 'CosinePrototypeProb':[95.2, 94.7, ..., 91.3], # Cosine Prototype Probing Performance

#             } 
#     """
#     il_mode = params.il_mode
#     assert il_mode in ['CIL','TIL'], 'NotImplemented for il_mode %s'%(il_mode)

#     num_task = CL_dataset.continual_config['NUM_TASK']
#     num_class = CL_dataset.continual_config['NUM_CLASS']
#     cur_num_class = CL_dataset.continual_config['CUR_NUM_CLASS']

#     # For Distributed Data Parallel
#     if hasattr(model,'module'):
#         model = model.module
#     else:
#         model = model
#     if isinstance(model,WrapModel):
#         model = model.model
#     if hasattr(model,'backbone_model'):
#         model = model.backbone_model

#     # Step 1: Compute all features
#     with torch.no_grad():
#         train_feature_list = []
#         train_label_idx_list = []
#         for t_id in range(num_task):
#             _train_feature_list = []
#             _train_label_idx_list = []
#             for lm_input in train_loader_list[t_id]:  
#                 extracted_features = obtain_features(params=params, 
#                                                         model=model, 
#                                                         lm_input=lm_input, 
#                                                         tokenizer=tokenizer)
#                 if il_mode == 'CIL':
#                     label_idx = lm_input['label_idx_cil']
#                 else:
#                     label_idx = lm_input['label_idx_til']
                
#                 extracted_features, label_idx = accelerator.gather_for_metrics((extracted_features, label_idx))
#                 _train_feature_list.append(extracted_features.detach().cpu())
#                 _train_label_idx_list.append(label_idx.cpu())

#             _train_feature_list = torch.cat(_train_feature_list, dim=0)
#             _train_label_idx_list = torch.cat(_train_label_idx_list, dim=0)

#             train_feature_list.append(_train_feature_list)
#             train_label_idx_list.append(_train_label_idx_list)

#         if il_mode == 'CIL':
#             train_features_all = torch.cat(train_feature_list, dim=0)
#             train_label_idx_all = torch.cat(train_label_idx_list, dim=0)
#         else:
#             train_features_all = train_feature_list
#             train_label_idx_all = train_label_idx_list
        
#         del train_feature_list, train_label_idx_list
#         accelerator.free_memory()

#     if accelerator.is_main_process: 
#         # Step 2: Fit the linear/coslinear/prototype/cosineprototype classifier

#         # Train on data of all tasks
#         if il_mode == 'CIL':

#             train_features = train_features_all
#             train_label_idx = train_label_idx_all

#             feature_dim = train_features[0].shape[-1]

#             prototype_layer = nn.Linear(in_features=feature_dim, out_features=num_class, bias=False).cuda()
#             cosprototype_layer = CosineLinear(in_features=feature_dim, out_features=num_class).cuda()

#             cur_class_prototypes = torch.zeros_like(prototype_layer.weight.data).cuda()
#             cnt_class_samples = {class_idx:0 for class_idx in range(num_class)}

#             linear_layer = nn.Linear(in_features=feature_dim, out_features=num_class, bias=False).cuda().train()
#             coslinear_layer = CosineLinear(in_features=feature_dim, out_features=num_class).cuda().train()

#             opt_linear = Adam(linear_layer.parameters(),lr=0.001)
#             opt_coslinear = Adam(coslinear_layer.parameters(),lr=0.001)
#             loss_fct = nn.CrossEntropyLoss()
#             num_chunk = train_features.shape[0]//128
#             loss_list_epochs_linear = []
#             loss_list_epochs_coslinear = []

#             for e_id in range(20): 

#                 loss_list_linear = []
#                 loss_list_coslinear = []
#                 shuffle_idx = list(range(train_features.shape[0]))
#                 shuffle(shuffle_idx)
#                 shuffle_idx = torch.tensor(shuffle_idx).to(train_features.device)

#                 for selected_idx in torch.chunk(shuffle_idx,num_chunk):
#                     _feature, _label_idx = train_features[selected_idx], train_label_idx[selected_idx]
#                     _feature, _label_idx = _feature.cuda(), _label_idx.cuda()

#                     if params.classification_type == 'word-level':
#                         label_mask = (_label_idx!=-100)
#                         _feature = _feature[label_mask] # flatten
#                         _label_idx = _label_idx[label_mask] # flatten

#                     # Train Linear Classifier
#                     logits_linear = linear_layer(_feature)
#                     loss_linear = loss_fct(logits_linear,_label_idx)
#                     opt_linear.zero_grad()
#                     loss_linear.backward()
#                     opt_linear.step()

#                     # Train Cosine Linear Classifier
#                     logits_coslinear = coslinear_layer(_feature)
#                     loss_coslinear = loss_fct(logits_coslinear,_label_idx)
#                     opt_coslinear.zero_grad()
#                     loss_coslinear.backward()
#                     opt_coslinear.step()

#                     loss_list_linear.append(loss_linear.item())
#                     loss_list_coslinear.append(loss_coslinear.item())

#                     # Compute Class Center for Prototype/CosinePrototype Classifier (Only need one epoch)
#                     if e_id==0:
#                         for class_idx in range(num_class):
#                             if class_idx not in _label_idx:
#                                 continue
#                             class_mask = (_label_idx==class_idx)
#                             cnt_class_samples[class_idx] += 1
#                             cur_class_prototypes[class_idx] += _feature[class_mask].detach().sum(dim=0)

#                 loss_list_epochs_linear.append(np.mean(loss_list_linear))
#                 loss_list_epochs_coslinear.append(np.mean(loss_list_coslinear))

#             # Set the weight of the Prototype/CosinePrototype Classifier
#             for class_idx in range(num_class):
#                 if cnt_class_samples[class_idx]==0:
#                     continue
#                 prototype_layer.weight.data[class_idx] = cur_class_prototypes[class_idx]/cnt_class_samples[class_idx]
#                 cosprototype_layer.weight.data[class_idx] = cur_class_prototypes[class_idx]/cnt_class_samples[class_idx]

#             del train_features, train_label_idx
#             torch.cuda.empty_cache()

#         # Train each classifier on the data of each task for il_mode=='TIL'
#         else:

#             feature_dim = train_features_all[0][0].shape[-1]

#             prototype_layer = nn.ModuleList([
#                 nn.Linear(in_features=feature_dim, out_features=cur_num_class[t_id], bias=False).cuda()
#                 for t_id in range(num_task)
#             ])
#             cosprototype_layer = nn.ModuleList([
#                 CosineLinear(in_features=feature_dim, out_features=cur_num_class[t_id]).cuda()
#                 for t_id in range(num_task)
#             ])

#             cur_class_prototypes = [
#                 torch.zeros_like(prototype_layer[t_id].weight.data).cuda()
#                 for t_id in range(num_task)
#             ]
#             cnt_class_samples = [
#                 {class_idx:0 for class_idx in range(cur_num_class[t_id])}
#                 for t_id in range(num_task)
#             ]

#             linear_layer = nn.ModuleList([
#                 nn.Linear(in_features=feature_dim, out_features=cur_num_class[t_id], bias=False).cuda().train()
#                 for t_id in range(num_task)
#             ])
#             coslinear_layer = nn.ModuleList([
#                 CosineLinear(in_features=feature_dim, out_features=cur_num_class[t_id]).cuda().train() 
#                 for t_id in range(num_task)
#             ])

#             opt_linear = Adam(linear_layer.parameters(),lr=0.001)
#             opt_coslinear = Adam(coslinear_layer.parameters(),lr=0.001)
#             loss_fct = nn.CrossEntropyLoss()

#             for t_id in range(num_task):

#                 train_features = train_features_all[t_id]
#                 train_label_idx = train_label_idx_all[t_id]

#                 num_chunk = train_features.shape[0]//128
#                 loss_list_epochs_linear = []
#                 loss_list_epochs_coslinear = []

#                 for e_id in range(20): 

#                     loss_list_linear = []
#                     loss_list_coslinear = []
#                     shuffle_idx = list(range(train_features.shape[0]))
#                     shuffle(shuffle_idx)
#                     shuffle_idx = torch.tensor(shuffle_idx).to(train_features.device)

#                     for selected_idx in torch.chunk(shuffle_idx,num_chunk):
#                         _feature, _label_idx = train_features[selected_idx], train_label_idx[selected_idx]
#                         _feature, _label_idx = _feature.cuda(), _label_idx.cuda()

#                         if params.classification_type == 'word-level':
#                             label_mask = (_label_idx!=-100)
#                             _feature = _feature[label_mask] # flatten
#                             _label_idx = _label_idx[label_mask] # flatten

#                         # Train Linear Classifier
#                         logits_linear = linear_layer[t_id](_feature)
#                         loss_linear = loss_fct(logits_linear,_label_idx)
#                         opt_linear.zero_grad()
#                         loss_linear.backward()
#                         opt_linear.step()

#                         # Train Cosine Linear Classifier
#                         logits_coslinear = coslinear_layer[t_id](_feature)
#                         loss_coslinear = loss_fct(logits_coslinear,_label_idx)
#                         opt_coslinear.zero_grad()
#                         loss_coslinear.backward()
#                         opt_coslinear.step()

#                         loss_list_linear.append(loss_linear.item())
#                         loss_list_coslinear.append(loss_coslinear.item())

#                         # Compute Class Center for Prototype/CosinePrototype Classifier (Only need one epoch)
#                         if e_id==0:
#                             for class_idx in range(cur_num_class[t_id]):
#                                 if class_idx not in _label_idx:
#                                     continue
#                                 class_mask = (_label_idx==class_idx)
#                                 cnt_class_samples[t_id][class_idx] += 1
#                                 cur_class_prototypes[t_id][class_idx] += _feature[class_mask].detach().sum(dim=0)

#                     loss_list_epochs_linear.append(np.mean(loss_list_linear))
#                     loss_list_epochs_coslinear.append(np.mean(loss_list_coslinear))

#                 # Set the weight of the Prototype/CosinePrototype Classifier
#                 for class_idx in range(cur_num_class[t_id]):
#                     if cnt_class_samples[t_id][class_idx]==0:
#                         continue
#                     prototype_layer[t_id].weight.data[class_idx] = cur_class_prototypes[t_id][class_idx]/cnt_class_samples[t_id][class_idx]
#                     cosprototype_layer[t_id].weight.data[class_idx] = cur_class_prototypes[t_id][class_idx]/cnt_class_samples[t_id][class_idx]

#                 del train_features, train_label_idx
#                 torch.cuda.empty_cache()

#             del train_features_all, train_label_idx_all

#         # Step 3: Evaluate the probing model for each tasks
#         with torch.no_grad():
#             prob_result_dict = {}
#             for metric_name, _classifier in zip(['LinearProb','CosineLinearProb','PrototypeProb','CosinePrototypeProb'],
#                                                 [linear_layer,coslinear_layer,prototype_layer,cosprototype_layer]):
#                 _classifier.eval()
#                 if params.classification_type == 'sentence-level':
#                     tasks_acc_list = []
#                     for t_id in range(num_task):
#                         acc_list = []
#                         for lm_input in test_loader_list[t_id]:  
#                             _feature = obtain_features(params=params, 
#                                                                 model=model, 
#                                                                 lm_input=lm_input, 
#                                                                 tokenizer=tokenizer)
#                             if il_mode == 'CIL':
#                                 _label_idx = lm_input['label_idx_cil']
#                                 _feature, _label_idx = accelerator.gather_for_metrics((_feature, _label_idx))
#                                 logits = _classifier(_feature)
#                             else:
#                                 _label_idx = lm_input['label_idx_til']
#                                 _feature, _label_idx = accelerator.gather_for_metrics((_feature, _label_idx))
#                                 logits = _classifier[t_id](_feature)
#                             preds = logits.argmax(dim=-1).detach().cpu()
#                             _label_idx = _label_idx.cpu()
#                             acc_list.extend([(_pred==_label).item() for _pred,_label in zip(preds,_label_idx)])
#                         tasks_acc_list.append(np.round(np.mean(acc_list)*100,3))
#                         torch.cuda.empty_cache()

#                 elif params.classification_type == 'word-level':
#                     tasks_acc_list = []
#                     # Only test once because the last test set contains all labels
#                     gold_lines = []
#                     pred_lines = []

#                     for lm_input in test_loader_list[num_task-1]: 
#                         _feature = obtain_features(params=params, 
#                                                     model=model, 
#                                                     lm_input=lm_input, 
#                                                     tokenizer=tokenizer)
#                         if il_mode == 'CIL':
#                             _label_idx = lm_input['label_idx_cil']
#                             _feature, _label_idx = accelerator.gather_for_metrics((_feature, _label_idx))
#                             logits = _classifier(_feature)
#                         else:
#                             raise NotImplementedError()
#                         preds = logits.argmax(dim=-1).detach().cpu()
#                         _label_idx = _label_idx.cpu()
#                         idx2label = CL_dataset.continual_config['idx2label']
#                         for _sent_gold, _sent_pred in zip(_label_idx,preds):
#                             gold_lines.append([idx2label[_gold.item()] for _gold, _pred in zip(_sent_gold,_sent_pred) if _gold.item()!=-100])
#                             pred_lines.append([idx2label[_pred.item()] for _gold, _pred in zip(_sent_gold,_sent_pred) if _gold.item()!=-100])
    
#                     # micro f1 (average over all instances)
#                     mi_f1 = f1_score(gold_lines, pred_lines)*100
#                     # macro f1 (default, average over each class)
#                     ma_f1 = f1_score(gold_lines, pred_lines, average='macro')*100
#                     logger.info('%s performance: macro_f1 = %.2f, micro_f1=%.2f'%(metric_name,ma_f1,mi_f1))

#                     tasks_acc_list = [ma_f1]*num_task

#                     torch.cuda.empty_cache()

#                 else:
#                     raise NotImplementedError()
                
#                 prob_result_dict[metric_name] = deepcopy(tasks_acc_list)

#         # save probing classifiers
#         if params.save_probing_classifiers:
#             torch.save(
#                 {
#                     'linear_layer': linear_layer.cpu(),
#                     'coslinear_layer': coslinear_layer.cpu(),
#                     'prototype_layer': prototype_layer.cpu(),
#                     'cosprototype_layer': cosprototype_layer.cpu(),
#                 }
#                 ,os.path.join(params.dump_path,'probing_classifiers_task%d.pth'%(task_id)))

#     else:
#         prob_result_dict = {
#             'LinearProb': [-1]*num_task,
#             'CosineLinearProb': [-1]*num_task,
#             'PrototypeProb': [-1]*num_task,
#             'CosinePrototypeProb': [-1]*num_task,
#         }

#     return prob_result_dict

def save_all_features_labels(params, save_name, train_loader_list, model, tokenizer, accelerator) -> None:
    '''
        Save all features and labels
    '''
    il_mode = params.il_mode
    assert il_mode in ['CIL','TIL'], 'NotImplemented for il_mode %s'%(il_mode)
    
    num_task = len(train_loader_list)
    with torch.no_grad():
        train_feature_list = []
        train_label_idx_list = []
        for t_id in range(num_task):
            _train_feature_list = []
            _train_label_idx_list = []
            for lm_input in train_loader_list[t_id]:  
                extracted_features = obtain_features(params=params, 
                                                        model=model, 
                                                        lm_input=lm_input, 
                                                        tokenizer=tokenizer)
                if params.il_mode == 'CIL':
                    label_idx = lm_input['label_idx_cil']
                else:
                    label_idx = lm_input['label_idx_til']
                
                extracted_features, label_idx = accelerator.gather_for_metrics((extracted_features, label_idx))
                _train_feature_list.append(extracted_features.detach().cpu())
                _train_label_idx_list.append(label_idx.cpu())

            _train_feature_list = torch.cat(_train_feature_list, dim=0)
            _train_label_idx_list = torch.cat(_train_label_idx_list, dim=0)

            train_feature_list.append(_train_feature_list)
            train_label_idx_list.append(_train_label_idx_list)

        if params.il_mode == 'CIL':
            train_features_all = torch.cat(train_feature_list, dim=0)
            train_label_idx_all = torch.cat(train_label_idx_list, dim=0)
        else:
            train_features_all = train_feature_list
            train_label_idx_all = train_label_idx_list

        save_path = os.path.join(params.dump_path,'train_features_labels_%s.pth'%(save_name))
        torch.save(
            {'features':train_features_all,'labels':train_label_idx_all},
            save_path
        )
        
        del train_feature_list, train_label_idx_list
        accelerator.free_memory()