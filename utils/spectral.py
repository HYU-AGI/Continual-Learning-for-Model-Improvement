import logging
import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from utils.wrapmodel import WrapModel
from utils.backbone import obtain_features

logger = logging.getLogger()

def evaluate_spectral_clustering_on_all_task(params, task_id, CL_dataset, test_loader_list, model, tokenizer, accelerator) -> dict:
    """
    모델의 특징을 사용하여 Spectral Clustering을 통해 클러스터링을 수행하고 성능을 평가합니다.
    분류기에 의존하지 않고 클러스터링 결과를 통해 모델의 성능을 측정합니다.

    Args:
        - params: 모델의 파라미터와 설정 정보
        - task_id: 현재 태스크 ID
        - CL_dataset: Continual Learning 데이터셋 객체
        - test_loader_list: 각 태스크의 테스트 데이터 로더 리스트
        - model: 평가할 모델
        - tokenizer: 모델과 연동된 토크나이저
        - accelerator: 분산 학습을 위한 가속기 객체

    Returns:
        - spectral_result_dict: 각 태스크별 클러스터링 정확도, ARI, NMI를 담은 딕셔너리
    """
    il_mode = params.il_mode
    assert il_mode in ['CIL', 'TIL'], 'NotImplemented for il_mode %s' % (il_mode)

    total_num_task = CL_dataset.continual_config['NUM_TASK']
    cur_num_class_list = CL_dataset.continual_config['CUR_NUM_CLASS']

    if il_mode == 'CIL':
        # CIL 모드에서는 전체 태스크의 데이터를 사용
        num_task = len(test_loader_list)  # test_loader_list의 실제 길이 (전체 태스크)
        cur_num_class = cur_num_class_list  # 모든 태스크의 클래스 수 리스트
        num_class = sum(cur_num_class)
    else:
        # TIL 모드에서는 현재까지의 태스크 데이터를 사용
        num_task = len(test_loader_list)  # 전달된 test_loader_list의 길이
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
        test_feature_list = []
        test_label_idx_list = []
        for t_id in range(num_task):
            _test_feature_list = []
            _test_label_idx_list = []
            for lm_input in test_loader_list[t_id]:
                extracted_features = obtain_features(params=params,
                                                     model=model,
                                                     lm_input=lm_input,
                                                     tokenizer=tokenizer)
                if il_mode == 'CIL':
                    label_idx = lm_input['label_idx_cil']
                else:
                    label_idx = lm_input['label_idx_til']

                extracted_features, label_idx = accelerator.gather_for_metrics((extracted_features, label_idx))
                _test_feature_list.append(extracted_features.detach().cpu())
                _test_label_idx_list.append(label_idx.cpu())

            _test_feature_list = torch.cat(_test_feature_list, dim=0)
            _test_label_idx_list = torch.cat(_test_label_idx_list, dim=0)

            test_feature_list.append(_test_feature_list)
            test_label_idx_list.append(_test_label_idx_list)

        if il_mode == 'CIL':
            test_features_all = torch.cat(test_feature_list, dim=0)
            test_label_idx_all = torch.cat(test_label_idx_list, dim=0)
        else:
            test_features_all = test_feature_list
            test_label_idx_all = test_label_idx_list

        del test_feature_list, test_label_idx_list
        torch.cuda.empty_cache()

    # Spectral Clustering 및 평가
    if accelerator.is_main_process:
        spectral_result_dict = {}
        if il_mode == 'CIL':
            # BFloat16 타입을 numpy가 지원하는 타입으로 변환
            features = test_features_all.float().numpy()
            labels = test_label_idx_all.numpy()
            num_clusters = sum(cur_num_class)
            print(f"CIL_num_data: {len(labels)}")
            print(f"CIL_num_clusters: {num_clusters}")

            # Spectral Clustering 수행
            spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
            cluster_assignments = spectral.fit_predict(features)

            # 클러스터 레이블과 실제 레이블 매핑
            contingency_matrix = metrics.cluster.contingency_matrix(labels, cluster_assignments)
            row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
            label_mapping = {col: row for row, col in zip(row_ind, col_ind)}
            mapped_clusters = np.vectorize(label_mapping.get)(cluster_assignments)

            # 전체 정확도, ARI, NMI 계산
            accuracy = np.mean(mapped_clusters == labels) * 100
            accuracy = np.round(accuracy, 3)
            ari = metrics.adjusted_rand_score(labels, mapped_clusters) * 100
            ari = np.round(ari, 4)
            nmi = metrics.normalized_mutual_info_score(labels, mapped_clusters) * 100
            nmi = np.round(nmi, 4)

            spectral_result_dict['Overall_Accuracy'] = accuracy
            spectral_result_dict['Overall_ARI'] = ari
            spectral_result_dict['Overall_NMI'] = nmi
            print(spectral_result_dict)

            # 클래스 레이블 범위 계산
            class_ranges = []
            start_label = 0
            for num_classes in cur_num_class_list:
                end_label = start_label + num_classes - 1
                class_ranges.append((start_label, end_label))
                start_label += num_classes

            # 각 태스크별로 정확도, ARI, NMI 계산
            for t_id in range(total_num_task):
                class_start, class_end = class_ranges[t_id]
                task_class_labels = np.arange(class_start, class_end + 1)
                task_mask = np.isin(test_label_idx_all.numpy(), task_class_labels)

                num_samples = task_mask.sum()
                if num_samples == 0:
                    task_accuracy = -1
                    task_ari = -1
                    task_nmi = -1
                else:
                    task_labels = labels[task_mask]
                    task_mapped_clusters = mapped_clusters[task_mask]
                    task_accuracy = np.mean(task_mapped_clusters == task_labels) * 100
                    task_accuracy = np.round(task_accuracy, 3)
                    task_ari = metrics.adjusted_rand_score(task_labels, task_mapped_clusters) * 100
                    task_ari = np.round(task_ari, 4)
                    task_nmi = metrics.normalized_mutual_info_score(task_labels, task_mapped_clusters) * 100
                    task_nmi = np.round(task_nmi, 4)
                spectral_result_dict[f'Task_{t_id}_Accuracy'] = task_accuracy
                spectral_result_dict[f'Task_{t_id}_ARI'] = task_ari
                spectral_result_dict[f'Task_{t_id}_NMI'] = task_nmi
                print(spectral_result_dict)
        else:
            # TIL 모드에서는 각 태스크별로 클러스터링 수행
            for t_id in range(total_num_task):
                if t_id >= num_task:
                    spectral_result_dict[f'Task_{t_id}_Accuracy'] = -1
                    spectral_result_dict[f'Task_{t_id}_ARI'] = -1
                    spectral_result_dict[f'Task_{t_id}_NMI'] = -1
                    continue

                features = test_features_all[t_id].float().numpy()
                labels = test_label_idx_all[t_id].numpy()
                num_clusters = cur_num_class[t_id]
                print(f"TIL_num_clusters: {num_clusters}")

                # Spectral Clustering 수행
                spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
                cluster_assignments = spectral.fit_predict(features)

                # 클러스터 레이블과 실제 레이블 매핑
                contingency_matrix = metrics.cluster.contingency_matrix(labels, cluster_assignments)
                row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
                label_mapping = {col: row for row, col in zip(row_ind, col_ind)}
                mapped_clusters = np.vectorize(label_mapping.get)(cluster_assignments)

                # 정확도, ARI, NMI 계산
                accuracy = np.mean(mapped_clusters == labels) * 100
                accuracy = np.round(accuracy, 3)
                ari = metrics.adjusted_rand_score(labels, mapped_clusters) * 100
                ari = np.round(ari, 4)
                nmi = metrics.normalized_mutual_info_score(labels, mapped_clusters) * 100
                nmi = np.round(nmi, 4)

                spectral_result_dict[f'Task_{t_id}_Accuracy'] = accuracy
                spectral_result_dict[f'Task_{t_id}_ARI'] = ari
                spectral_result_dict[f'Task_{t_id}_NMI'] = nmi
                print(spectral_result_dict)

    else:
        # 메인 프로세스가 아닌 경우 결과를 -1로 채움
        num_tasks_for_results = total_num_task
        spectral_result_dict = {}
        for t_id in range(num_tasks_for_results):
            spectral_result_dict[f'Task_{t_id}_Accuracy'] = -1
            spectral_result_dict[f'Task_{t_id}_ARI'] = -1
            spectral_result_dict[f'Task_{t_id}_NMI'] = -1
        if il_mode == 'CIL':
            spectral_result_dict['Overall_Accuracy'] = -1
            spectral_result_dict['Overall_ARI'] = -1
            spectral_result_dict['Overall_NMI'] = -1

    return spectral_result_dict
