# import os
# import numpy as np
# import logging
# import torch
# from abc import abstractmethod
# from copy import deepcopy
# from accelerate.utils import gather_object


# from utils.evaluation import compute_backward_transfer, compute_forward_transfer, compute_average_acc, compute_average_inc_acc, compute_forgetting
# from utils.probing import probing_on_all_task, save_all_features_labels

# logger = logging.getLogger()

# class BaseLearner(object):
#     def __init__(self, params, CL_dataset, accelerator):
#         # parameters
#         self.params = params
#         self.best_model_ckpt_name = None
#         self.global_step=0
#         self.CL_dataset = CL_dataset
#         self.accelerator = accelerator

#         # initialization
#         self.model = None
#         self.classifier = None
#         self.optimizer = None
#         self.train_loader_list, self.dev_loader_list, self.test_loader_list = [], [], []
#         self.build_metric()
#         self.build_backbone()
#         self.build_classifier()
#         self.build_optimizer()
#         self.build_dataloader()
#         self.build_buffer()
#         self.accelerate_prepare()
        
#     # ================================= Prepare model, data and optimizer =======================================
#     @abstractmethod
#     def build_metric(self):
#         pass

#     @abstractmethod
#     def build_backbone(self):
#         pass

#     @abstractmethod
#     def build_classifier(self):
#         pass

#     @abstractmethod
#     def build_optimizer(self):
#         pass
        
#     @abstractmethod
#     def build_dataloader(self):
#         pass

#     @abstractmethod
#     def build_buffer(self):
#         pass

#     @abstractmethod
#     def accelerate_prepare(self):
#         pass
#     # ==============================================================================================

#     # ================================= Task-Level Functions =======================================
#     def incremental_training(self):

#         num_task = self.CL_dataset.continual_config['NUM_TASK']

#         # Learn Tasks Incrementally
#         for task_id in range(num_task):
#             if self.accelerator.is_main_process: 
#                 logger.info("============================================================================")   
#                 logger.info("Beggin training the task %d (total %d tasks)"%(task_id+1, num_task))     
#                 logger.info("============================================================================")
#             self.begin_task(task_id)
#             self.train_epochs(task_id)
#             self.end_task(task_id)  

#     def begin_task(self, task_id):
        
#         self.best_score = -1
#         self.step = 0

#         # save features
#         if task_id==0 and self.params.save_features_before_after_IL:
#             save_all_features_labels(self.params, 
#                                      'BeforeIL',
#                                      self.train_loader_list, 
#                                      self.model, 
#                                      self.tokenizer, 
#                                      self.accelerator)
        
#     def end_task(self, task_id):

#         # testing
#         if self.accelerator.is_main_process:
#             logger.info("Testing...")

#         result_dict = self.evaluate_model(task_id=task_id) 
#         il_mode = self.params.il_mode      
#         for t_id, _acc in enumerate(result_dict['Test_Acc_List']):
#             self.result_summary.update(task_id, t_id, _acc)
#         if self.accelerator.is_main_process:
#             logger.info('Mode = %s, Result Summary Test After Task %d = \n%s'%(il_mode, 
#                                                                         task_id,
#                                                                         self.result_summary.print_format()))
#         if il_mode == 'IIL' and self.params.method != 'ICL':
#             for t_id, _acc in enumerate(result_dict['Train_Acc_List']):
#                 self.result_summary_train.update(task_id, t_id, _acc)
#             if self.accelerator.is_main_process:
#                 logger.info('Mode = %s, Result Summary Train After Task %d = \n%s'%(il_mode, 
#                                                                             task_id,
#                                                                             self.result_summary_train.print_format()))

#         # save ckpt
#         if self.params.save_ckpt:
#             torch.save(
#                 {
#                     'model': deepcopy(self.model).cpu(),
#                     'classifier_list': deepcopy(self.classifier_list).cpu(),
#                 }
#                 ,os.path.join(self.params.dump_path,'last_ckpt_task%d.pth'%(task_id)))
            
#         # save features
#         num_task = self.CL_dataset.continual_config['NUM_TASK']
#         if task_id==num_task-1 and self.params.save_features_before_after_IL:
#             save_all_features_labels(self.params, 
#                                      'AfterIL',
#                                      self.train_loader_list, 
#                                      self.model, 
#                                      self.tokenizer, 
#                                      self.accelerator)

#         # Save GPU memory
#         torch.cuda.empty_cache()
#     # ===========================================================================================


#     # ================================= Epoch-Level Functions ====================================
#     @abstractmethod
#     def train_epochs(self, task_id):
#         pass
#     # ===========================================================================================


#     # ================== Evaluation, Logging, Saving and Loading Functions ======================
#     def finish_training(self):
#         '''
#             Finish training: print the result
#         '''
#         log_dict = {}
#         il_mode = self.params.il_mode
#         if self.accelerator.is_main_process:
#             logger.info('Mode = %s, Summary Test Acc = \n%s'%(il_mode,self.result_summary.print_format()))
#         # Compute Forward and Backward Transfer according to Result Summary for the whole learning process
#         bwt_acc = compute_backward_transfer(self.result_summary.get_value())
#         fwt_acc = compute_forward_transfer(self.result_summary.get_value()) 
#         fgt_acc = compute_forgetting(self.result_summary.get_value()) 
#         aver_acc = compute_average_acc(self.result_summary.get_value()) 
#         aver_inc_acc = compute_average_inc_acc(self.result_summary.get_value()) 
#         log_dict['Test_Aver_ACC'] = aver_acc
#         log_dict['Test_Bwt_ACC'] = bwt_acc
#         log_dict['Test_Fwt_ACC'] = fwt_acc
#         log_dict['Test_Fgt_ACC'] = fgt_acc
#         log_dict['Test_Aver_Inc_ACC'] = aver_inc_acc

#         if il_mode == 'IIL' and self.params.method != 'ICL':
#             if self.accelerator.is_main_process:
#                 logger.info('Mode = %s, Summary Train Acc = \n%s'%(il_mode,self.result_summary_train.print_format()))
#             # Compute Forward and Backward Transfer according to Result Summary for the whole learning process
#             bwt_acc = compute_backward_transfer(self.result_summary_train.get_value())
#             fwt_acc = compute_forward_transfer(self.result_summary.get_value())
#             fgt_acc = compute_forgetting(self.result_summary_train.get_value())
#             aver_acc = compute_average_acc(self.result_summary_train.get_value())
#             aver_inc_acc = compute_average_inc_acc(self.result_summary_train.get_value())
#             log_dict['Train_Aver_ACC'] = aver_acc
#             log_dict['Train_Bwt_ACC'] = bwt_acc
#             log_dict['Test_Fwt_ACC'] = fwt_acc
#             log_dict['Train_Fgt_ACC'] = fgt_acc
#             log_dict['Train_Aver_Inc_ACC'] = aver_inc_acc

#         if self.accelerator.is_main_process:
#             logger.info('Mode = %s, Summary Result = \n%s'%(il_mode,log_dict))    
#         self.accelerator.log(log_dict,step=self.global_step)
#         print(self.result_summary.get_value())
    
#         # Delete checkpoints
#         # if not self.params.save_ckpt:
#         #     for file_name in os.listdir(self.params.dump_path):
#         #         if file_name[-4:] == '.pth':
#         #             os.remove(os.path.join(self.params.dump_path,file_name))

#         # End Wandb
#         self.accelerator.end_training()

#     def evaluate_model(self, task_id: int) -> dict:
#         '''
#         Evaluate the model and log the result

#         Args:
#             - task_id: task_id records how many tasks the model have learned previously

#         Return:
#             - {

#                 'Test_Acc_List':[90.42, 92.19, ..., 87.54], # Observed result
                
#                 'LinearProb_List':[95.2, 94.7, ..., 91.3],          # Linear Probing Performance

#                 'CosineLinearProb_List':[95.2, 94.7, ..., 91.3],    # Cosine Linear Probing Performance

#                 'PrototypeProb_List':[95.2, 94.7, ..., 91.3],       # Prototype Probing Performance

#                 'CosinePrototypeProb_List':[95.2, 94.7, ..., 91.3], # Cosine Prototype Probing Performance

#             }
#         '''
#         result_dict = {}
#         log_dict = {}

#         # NOTE: 
#         # When using classifier, we can only evaluate the model on the tasks which has been learned
#         # The reason is that the classifiers of unseen tasks is randomly initialized and the result is meaningless.
#         # For example, it is meaningless to let a CIL model which has learned 50 classes to make predictions in 150 classes.
#         cur_task_id = task_id
#         il_mode = self.params.il_mode
        
#         acc_list = self.evaluate_all_seen_task_tc(cur_task_id, 'test', il_mode)
#         result_dict['Test_Acc_List'] = acc_list
#         for t_id in range(cur_task_id+1):
#             log_dict['Test_Acc_Task_%d'%(t_id)] = acc_list[t_id]
#         log_dict['Test_Acc_Task_Seen'] = np.round(np.mean(acc_list[:cur_task_id+1]),3)
#         if self.params.classifier=='None':
#             log_dict['Test_Acc_Task_All'] = np.round(np.mean(acc_list),3)
#         if self.accelerator.is_main_process:
#             logger.info('Mode = %s, Test Result = %s'%(il_mode, log_dict))
        
#         # Additional Evaluation on training set for Instance Incremental Learning
#         if il_mode == 'IIL' and self.params.method != 'ICL':
#             acc_list = self.evaluate_all_seen_task_tc(cur_task_id, 'train', il_mode)
#             result_dict['Train_Acc_List'] = acc_list
#             for t_id in range(cur_task_id+1):
#                 log_dict['Train_Acc_Task_%d'%(t_id)]=acc_list[t_id]
#             log_dict['Train_Acc_Task_Seen'] = np.round(np.mean(acc_list[:cur_task_id+1]),3)
#             if self.params.classifier=='None':
#                 log_dict['Train_Acc_Task_All'] = np.round(np.mean(acc_list),3)
#             if self.accelerator.is_main_process:
#                 logger.info('Mode = %s, Train Result = %s'%(il_mode, log_dict))

#         self.accelerator.log(log_dict,step=self.global_step)

#         if self.params.is_probing:
#             prob_acc_dict = probing_on_all_task(
#                 params=self.params,
#                 task_id=task_id,
#                 CL_dataset=self.CL_dataset, 
#                 train_loader_list=self.train_loader_list, 
#                 test_loader_list=self.test_loader_list, 
#                 model=self.model, 
#                 tokenizer=self.tokenizer, 
#                 accelerator=self.accelerator
#             )
#             prob_log_dict = {}
#             for prob_metric in prob_acc_dict.keys():
#                 result_dict['%s_List'%(prob_metric)] = prob_acc_dict[prob_metric]
#                 prob_log_dict['%s_Acc_Task_Seen'%(prob_metric)] = np.round(np.mean(prob_acc_dict[prob_metric][:cur_task_id+1]),3)
#                 prob_log_dict['%s_Acc_Task_All'%(prob_metric)] = np.round(np.mean(prob_acc_dict[prob_metric]),3)
#             if self.accelerator.is_main_process:
#                 logger.info('Probing Result = %s'%(prob_log_dict))
#             self.accelerator.log(prob_log_dict, step=self.global_step)

#         return result_dict

#     def evaluate_all_seen_task_tc(self,
#                                 cur_task_id: int, 
#                                 phase: str,
#                                 il_mode: str) -> list:
#         '''
#             Evaluate the model on all seen tasks

#             Params: 
#                 - cur_task_id: cur_task_id records how many tasks the model have learned previously
#                 - phase: 'train','dev'or'test'
#                 - il_mode: 'CIL' or 'TIL'

#             Return:
#                 - {
#                     [90.42, 92.19, ..., 87.54], 
#                 }: representing the accuracy of tasks [0, 1, ..., cur_task_id].

#         '''
#         assert phase in ['train','test','dev']

#         acc_list = []

#         # Evaluate on all seen tasks
#         if self.params.classification_type == 'sentence-level':

#             save_dict_all = None

#             for eval_t_id in range(cur_task_id+1):

#                 if il_mode=='IIL':
#                     acc, save_dict = self.evaluate_current_task(eval_t_id, cur_task_id, phase, il_mode)
#                     if save_dict_all is not None:
#                         for k,v in save_dict.items():
#                             save_dict_all[k] = v
#                     else:
#                         save_dict_all = save_dict
#                 else:
#                     acc = self.evaluate_current_task(eval_t_id, cur_task_id, phase, il_mode)

#                 acc_list.append(acc)

#             save_dict_all = [save_dict_all] # transform to list, otherwise gather_object() will not collect correctly

#             gathered_save_dict_all = gather_object(save_dict_all)

#             if self.accelerator.is_main_process:
#                 if gathered_save_dict_all is not None:
#                     with open(os.path.join(self.params.dump_path,
#                                         '%s_cur_task_%d_save_result.npy'%(phase,cur_task_id)),'wb') as f:
#                         np.save(f,gathered_save_dict_all)

#         # Evaluate on all seen tasks
#         # NOTE: We only test once because the test set of task task_id contains all seen labels from task 0 - task task_id)
#         elif self.params.classification_type == 'word-level':

#             eval_t_id = cur_task_id

#             acc = self.evaluate_current_task(eval_t_id, cur_task_id, phase, il_mode)

#             acc_list = [acc]*(cur_task_id+1)

#         return acc_list 

#     @abstractmethod
#     def evaluate_current_task(self,
#                                 eval_task_id: int, 
#                                 cur_task_id: int, 
#                                 phase: str,
#                                 il_mode: str) -> float:
#         '''
#             Evaluate the model on the current task

#             Params: 
#                 - eval_task_id: the id of the task to be evaluated, 
#                 this information should NOT be provided to the CIL model during inference!
#                 - cur_task_id: the id recording how many tasks the model has learned,
#                 this information can be provided to the CIL model during inference.
#                 - phase: 'train','dev'or'test'
#                 - il_mode: 'CIL' or 'TIL'

#             Return:
#                 - acc: CIL accuracy (%) or 'TIL': TIL accuracy (%)
#         '''
#         pass
#     # ===========================================================================================

import os
import numpy as np
import pandas as pd
import logging
import torch
from abc import abstractmethod
from copy import deepcopy
from accelerate.utils import gather_object

from utils.evaluation import compute_backward_transfer, compute_forward_transfer, compute_average_acc, compute_average_inc_acc, compute_forgetting
from utils.probing import probing_on_all_task, save_all_features_labels
from utils.kmeans import evaluate_kmeans_on_all_task
from utils.gmm import evaluate_gmm_on_all_task
from utils.spectral import evaluate_spectral_clustering_on_all_task
from utils.agglomerative import evaluate_agglomerative_clustering_on_all_task
from utils.deep_clustering import evaluate_deep_clustering_on_all_task

logger = logging.getLogger()

class BaseLearner(object):
    def __init__(self, params, CL_dataset, accelerator):
        # parameters
        self.params = params
        self.best_model_ckpt_name = None
        self.global_step = 0
        self.CL_dataset = CL_dataset
        self.accelerator = accelerator

        # initialization
        self.model = None
        self.classifier = None
        self.optimizer = None
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = [], [], []
        self.build_metric()
        self.build_backbone()
        self.build_classifier()
        self.build_optimizer()
        self.build_dataloader()
        self.build_buffer()
        self.accelerate_prepare()

    # ================================= Prepare model, data and optimizer =======================================
    @abstractmethod
    def build_metric(self):
        pass

    @abstractmethod
    def build_backbone(self):
        pass

    @abstractmethod
    def build_classifier(self):
        pass

    @abstractmethod
    def build_optimizer(self):
        pass
        
    @abstractmethod
    def build_dataloader(self):
        pass

    @abstractmethod
    def build_buffer(self):
        pass

    @abstractmethod
    def accelerate_prepare(self):
        pass
    # ==============================================================================================

    # ================================= Task-Level Functions =======================================
    def incremental_training(self):

        num_task = self.CL_dataset.continual_config['NUM_TASK']

        # Learn Tasks Incrementally
        for task_id in range(num_task):
            if self.accelerator.is_main_process: 
                logger.info("============================================================================")   
                logger.info("Beggin training the task %d (total %d tasks)" % (task_id + 1, num_task))     
                logger.info("============================================================================")
            self.begin_task(task_id)
            self.train_epochs(task_id)
            self.end_task(task_id)  

    def begin_task(self, task_id):
        
        self.best_score = -1
        self.step = 0

        # save features
        if task_id == 0 and self.params.save_features_before_after_IL:
            save_all_features_labels(self.params, 
                                     'BeforeIL',
                                     self.train_loader_list, 
                                     self.model, 
                                     self.tokenizer, 
                                     self.accelerator)
        
    def end_task(self, task_id):

        # testing
        if self.accelerator.is_main_process:
            logger.info("Testing...")

        result_dict = self.evaluate_model(task_id=task_id)
        il_mode = self.params.il_mode

        # Update current task result
        for t_id, _acc in enumerate(result_dict['Test_Acc_List']):
            self.result_summary.update(task_id, t_id, _acc)

        # Update next task result if available
        if 'Next_Acc_List' in result_dict:
            for t_id, _acc_next in enumerate(result_dict['Next_Acc_List']):
                self.result_summary.update(task_id, t_id + 1, _acc_next)

        if self.accelerator.is_main_process:
            logger.info('Mode = %s, Result Summary Test After Task %d = \n%s' % (il_mode, 
                                                                                 task_id,
                                                                                 self.result_summary.print_format()))

        # Save checkpoint
        if self.params.save_ckpt:
            torch.save(
                {
                    'model': deepcopy(self.model).cpu(),
                    'classifier_list': deepcopy(self.classifier_list).cpu(),
                }
                , os.path.join(self.params.dump_path, 'last_ckpt_task%d.pth' % (task_id)))

        # Save features
        num_task = self.CL_dataset.continual_config['NUM_TASK']
        if task_id == num_task - 1 and self.params.save_features_before_after_IL:
            save_all_features_labels(self.params, 
                                     'AfterIL',
                                     self.train_loader_list, 
                                     self.model, 
                                     self.tokenizer, 
                                     self.accelerator)

        # Save GPU memory
        torch.cuda.empty_cache()
    # ===========================================================================================


    # ================== Evaluation, Logging, Saving and Loading Functions ======================
    # def evaluate_model(self, task_id: int) -> dict:
    #     '''
    #     Evaluate the model and log the result

    #     Args:
    #         - task_id: task_id records how many tasks the model have learned previously

    #     Return:
    #         - result_dict: dictionary containing Test_Acc_List and optionally Next_Acc_List
    #     '''
    #     result_dict = {}
    #     log_dict = {}

    #     cur_task_id = task_id
    #     il_mode = self.params.il_mode

    #     # Evaluate current task and next task (if applicable)
    #     acc_list, acc_next_list = self.evaluate_all_seen_task_tc(cur_task_id, 'test', il_mode)
    #     result_dict['Test_Acc_List'] = acc_list

    #     # Log current task accuracy
    #     for t_id in range(cur_task_id + 1):
    #         log_dict['Test_Acc_Task_%d' % (t_id)] = acc_list[t_id]

    #     # Store next task result if applicable
    #     if acc_next_list is not None:
    #         result_dict['Next_Acc_List'] = acc_next_list
    #         log_dict['Test_Acc_Task_%d' % (cur_task_id + 1)] = np.round(np.mean(acc_next_list), 3)

    #     # Log results
    #     log_dict['Test_Acc_Task_Seen'] = np.round(np.mean(acc_list[:cur_task_id + 1]), 3)
    #     if self.params.classifier == 'None':
    #         log_dict['Test_Acc_Task_All'] = np.round(np.mean(acc_list), 3)

    #     if self.accelerator.is_main_process:
    #         logger.info('Mode = %s, Test Result = %s' % (il_mode, log_dict))
    #     self.accelerator.log(log_dict, step=self.global_step)

    #     return result_dict
    
    def evaluate_model(self, task_id: int) -> dict:
        '''
        모델을 평가하고 결과를 로그로 기록합니다.

        Args:
            - task_id: 모델이 이전에 학습한 태스크의 수를 나타내는 태스크 ID

        Return:
            - result_dict: Test_Acc_List 및 필요한 경우 Next_Acc_List를 포함하는 딕셔너리
        '''
        result_dict = {}
        log_dict = {}

        cur_task_id = task_id
        il_mode = self.params.il_mode

        # 현재 태스크와 이전에 학습한 태스크 평가
        acc_list, acc_next_list = self.evaluate_all_seen_task_tc(cur_task_id, 'test', il_mode)
        result_dict['Test_Acc_List'] = acc_list

        # 현재 태스크의 정확도 로그에 기록
        for t_id in range(cur_task_id + 1):
            log_dict['Test_Acc_Task_%d' % (t_id)] = acc_list[t_id]

        # 결과 요약 업데이트 (기본 평가 성능)
        logger.info(f'Updating task {cur_task_id}, eval_task_id {cur_task_id}, with value {acc_list[cur_task_id]}')
        self.result_summary.update(cur_task_id, cur_task_id, acc_list[cur_task_id])  # 현재 태스크 업데이트

        # acc_next_list가 존재하고 None이 아닌지 확인
        if acc_next_list is not None and cur_task_id + 1 < len(self.result_summary.result_summary):
            # acc_next_list[cur_task_id]가 유효한지 확인 후 업데이트
            if acc_next_list[cur_task_id] is not None:
                logger.info(f'Updating next task: task {cur_task_id}, eval_task_id {cur_task_id + 1}, with value {acc_next_list[cur_task_id]}')
                self.result_summary.update(cur_task_id, cur_task_id + 1, acc_next_list[cur_task_id])  # 다음 태스크 업데이트

        # 결과 로그 기록
        log_dict['Test_Acc_Task_Seen'] = np.round(np.mean(acc_list[:cur_task_id + 1]), 3)
        if self.params.classifier == 'None':
            log_dict['Test_Acc_Task_All'] = np.round(np.mean(acc_list), 3)

        if self.accelerator.is_main_process:
            logger.info('Mode = %s, Test Result = %s' % (il_mode, log_dict))
            logger.info(f'Result Summary Test After Task {cur_task_id} =\n{self.result_summary.print_format()}')
        self.accelerator.log(log_dict, step=self.global_step)

        # # K-Means 클러스터링 평가
        # kmeans_result_dict = evaluate_kmeans_on_all_task(
        #     params=self.params,
        #     task_id=task_id,
        #     CL_dataset=self.CL_dataset,
        #     test_loader_list=self.test_loader_list,
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     accelerator=self.accelerator
        # )

        # # K-Means 결과 로그 및 업데이트
        # total_num_task = self.CL_dataset.continual_config['NUM_TASK']

        # # 각 태스크별로 결과를 저장할 리스트 초기화
        # kmeans_acc_list = []
        # kmeans_ari_list = []
        # kmeans_nmi_list = []

        # for t_id in range(total_num_task):
        #     acc = kmeans_result_dict.get(f'Task_{t_id}_Accuracy', -1)
        #     ari = kmeans_result_dict.get(f'Task_{t_id}_ARI', -1)
        #     nmi = kmeans_result_dict.get(f'Task_{t_id}_NMI', -1)
        #     kmeans_acc_list.append(acc)
        #     kmeans_ari_list.append(ari)
        #     kmeans_nmi_list.append(nmi)

        # # 결과를 result_dict에 저장
        # result_dict['KMeans_Acc_List'] = kmeans_acc_list
        # result_dict['KMeans_ARI_List'] = kmeans_ari_list
        # result_dict['KMeans_NMI_List'] = kmeans_nmi_list

        # # 평균 계산을 위한 함수 정의
        # def compute_mean(values):
        #     valid_values = [v for v in values if v != -1]
        #     return np.round(np.mean(valid_values), 4) if valid_values else -1

        # # 현재까지 학습한 태스크와 전체 태스크에 대한 평균 계산
        # kmeans_log_dict = {
        #     'KMeans_Acc_Task_Seen': compute_mean(kmeans_acc_list[:task_id + 1]),
        #     'KMeans_Acc_Task_All': compute_mean(kmeans_acc_list),
        #     'KMeans_ARI_Task_Seen': compute_mean(kmeans_ari_list[:task_id + 1]),
        #     'KMeans_ARI_Task_All': compute_mean(kmeans_ari_list),
        #     'KMeans_NMI_Task_Seen': compute_mean(kmeans_nmi_list[:task_id + 1]),
        #     'KMeans_NMI_Task_All': compute_mean(kmeans_nmi_list),
        # }

        # # CIL 모드의 경우 전체 결과 추가
        # if self.params.il_mode == 'CIL':
        #     overall_acc = kmeans_result_dict.get('Overall_Accuracy', -1)
        #     overall_ari = kmeans_result_dict.get('Overall_ARI', -1)
        #     overall_nmi = kmeans_result_dict.get('Overall_NMI', -1)
        #     kmeans_log_dict['KMeans_Overall_Accuracy'] = overall_acc
        #     kmeans_log_dict['KMeans_Overall_ARI'] = overall_ari
        #     kmeans_log_dict['KMeans_Overall_NMI'] = overall_nmi

        # # 각 태스크의 K-Means 결과를 업데이트
        # for t_id in range(total_num_task):
        #     acc = kmeans_acc_list[t_id]
        #     ari = kmeans_ari_list[t_id]
        #     nmi = kmeans_nmi_list[t_id]
        #     if acc != -1:
        #         logger.info(f'Updating KMeans result: task {task_id}, eval_task_id {t_id}, with Accuracy {acc}, ARI {ari}, NMI {nmi}')
        #         self.kmeans_result_summary.update(task_id, t_id, Accuracy=acc, ARI=ari, NMI=nmi)

        # # K-Means 결과 요약 출력
        # if self.accelerator.is_main_process:
        #     logger.info(f'Result Summary KMeans After Task {task_id} =\n{self.kmeans_result_summary.print_format()}')
        #     logger.info('KMeans Result = %s' % (kmeans_log_dict))

        # self.accelerator.log(kmeans_log_dict, step=self.global_step)


        # # GMM 클러스터링 평가
        # gmm_result_dict = evaluate_gmm_on_all_task(
        #     params=self.params,
        #     task_id=task_id,
        #     CL_dataset=self.CL_dataset,
        #     test_loader_list=self.test_loader_list,
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     accelerator=self.accelerator
        # )

        # # 결과 처리
        # total_num_task = self.CL_dataset.continual_config['NUM_TASK']
        # gmm_acc_list = []
        # gmm_ari_list = []
        # gmm_nmi_list = []

        # for t_id in range(total_num_task):
        #     acc = gmm_result_dict.get(f'Task_{t_id}_Accuracy', -1)
        #     ari = gmm_result_dict.get(f'Task_{t_id}_ARI', -1)
        #     nmi = gmm_result_dict.get(f'Task_{t_id}_NMI', -1)
        #     gmm_acc_list.append(acc)
        #     gmm_ari_list.append(ari)
        #     gmm_nmi_list.append(nmi)

        # # 결과를 result_dict에 저장
        # result_dict['GMM_Acc_List'] = gmm_acc_list
        # result_dict['GMM_ARI_List'] = gmm_ari_list
        # result_dict['GMM_NMI_List'] = gmm_nmi_list

        # # 평균 계산
        # def compute_mean(values):
        #     valid_values = [v for v in values if v != -1]
        #     return np.round(np.mean(valid_values), 4) if valid_values else -1

        # gmm_log_dict = {
        #     'GMM_Acc_Task_Seen': compute_mean(gmm_acc_list[:task_id + 1]),
        #     'GMM_Acc_Task_All': compute_mean(gmm_acc_list),
        #     'GMM_ARI_Task_Seen': compute_mean(gmm_ari_list[:task_id + 1]),
        #     'GMM_ARI_Task_All': compute_mean(gmm_ari_list),
        #     'GMM_NMI_Task_Seen': compute_mean(gmm_nmi_list[:task_id + 1]),
        #     'GMM_NMI_Task_All': compute_mean(gmm_nmi_list),
        # }

        # # CIL 모드의 경우 전체 결과 추가
        # if self.params.il_mode == 'CIL':
        #     overall_acc = gmm_result_dict.get('Overall_Accuracy', -1)
        #     overall_ari = gmm_result_dict.get('Overall_ARI', -1)
        #     overall_nmi = gmm_result_dict.get('Overall_NMI', -1)
        #     gmm_log_dict['GMM_Overall_Accuracy'] = overall_acc
        #     gmm_log_dict['GMM_Overall_ARI'] = overall_ari
        #     gmm_log_dict['GMM_Overall_NMI'] = overall_nmi

        # # 결과 로그 및 업데이트
        # for t_id in range(total_num_task):
        #     acc = gmm_acc_list[t_id]
        #     ari = gmm_ari_list[t_id]
        #     nmi = gmm_nmi_list[t_id]
        #     if acc != -1:
        #         logger.info(f'Updating GMM result: task {task_id}, eval_task_id {t_id}, with Accuracy {acc}, ARI {ari}, NMI {nmi}')
        #         self.gmm_result_summary.update(task_id, t_id, Accuracy=acc, ARI=ari, NMI=nmi)

        # # 결과 요약 출력
        # if self.accelerator.is_main_process:
        #     logger.info(f'Result Summary GMM After Task {task_id} =\n{self.gmm_result_summary.print_format()}')
        #     logger.info('GMM Result = %s' % (gmm_log_dict))

        # self.accelerator.log(gmm_log_dict, step=self.global_step)
        
        # Spectral Clustering 평가
        spectral_result_dict = evaluate_spectral_clustering_on_all_task(
            params=self.params,
            task_id=task_id,
            CL_dataset=self.CL_dataset,
            test_loader_list=self.test_loader_list,
            model=self.model,
            tokenizer=self.tokenizer,
            accelerator=self.accelerator
        )

        # 결과 처리
        total_num_task = self.CL_dataset.continual_config['NUM_TASK']
        spectral_acc_list = []
        spectral_ari_list = []
        spectral_nmi_list = []

        for t_id in range(total_num_task):
            acc = spectral_result_dict.get(f'Task_{t_id}_Accuracy', -1)
            ari = spectral_result_dict.get(f'Task_{t_id}_ARI', -1)
            nmi = spectral_result_dict.get(f'Task_{t_id}_NMI', -1)
            spectral_acc_list.append(acc)
            spectral_ari_list.append(ari)
            spectral_nmi_list.append(nmi)

        # 결과를 result_dict에 저장
        result_dict['Spectral_Acc_List'] = spectral_acc_list
        result_dict['Spectral_ARI_List'] = spectral_ari_list
        result_dict['Spectral_NMI_List'] = spectral_nmi_list

        # 평균 계산
        def compute_mean(values):
            valid_values = [v for v in values if v != -1]
            return np.round(np.mean(valid_values), 4) if valid_values else -1

        spectral_log_dict = {
            'Spectral_Acc_Task_Seen': compute_mean(spectral_acc_list[:task_id + 1]),
            'Spectral_Acc_Task_All': compute_mean(spectral_acc_list),
            'Spectral_ARI_Task_Seen': compute_mean(spectral_ari_list[:task_id + 1]),
            'Spectral_ARI_Task_All': compute_mean(spectral_ari_list),
            'Spectral_NMI_Task_Seen': compute_mean(spectral_nmi_list[:task_id + 1]),
            'Spectral_NMI_Task_All': compute_mean(spectral_nmi_list),
        }

        # CIL 모드의 경우 전체 결과 추가
        if self.params.il_mode == 'CIL':
            overall_acc = spectral_result_dict.get('Overall_Accuracy', -1)
            overall_ari = spectral_result_dict.get('Overall_ARI', -1)
            overall_nmi = spectral_result_dict.get('Overall_NMI', -1)
            spectral_log_dict['Spectral_Overall_Accuracy'] = overall_acc
            spectral_log_dict['Spectral_Overall_ARI'] = overall_ari
            spectral_log_dict['Spectral_Overall_NMI'] = overall_nmi

        # 결과 로그 및 업데이트
        for t_id in range(total_num_task):
            acc = spectral_acc_list[t_id]
            ari = spectral_ari_list[t_id]
            nmi = spectral_nmi_list[t_id]
            if acc != -1:
                logger.info(f'Updating Spectral Clustering result: task {task_id}, eval_task_id {t_id}, with Accuracy {acc}, ARI {ari}, NMI {nmi}')
                self.spectral_result_summary.update(task_id, t_id, Accuracy=acc, ARI=ari, NMI=nmi)

        # 결과 요약 출력
        if self.accelerator.is_main_process:
            logger.info(f'Result Summary Spectral Clustering After Task {task_id} =\n{self.spectral_result_summary.print_format()}')
            logger.info('Spectral Clustering Result = %s' % (spectral_log_dict))

        self.accelerator.log(spectral_log_dict, step=self.global_step)

        # # Agglomerative Clustering 평가
        # agglomerative_result_dict = evaluate_agglomerative_clustering_on_all_task(
        #     params=self.params,
        #     task_id=task_id,
        #     CL_dataset=self.CL_dataset,
        #     test_loader_list=self.test_loader_list,
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     accelerator=self.accelerator
        # )

        # # 결과 처리
        # total_num_task = self.CL_dataset.continual_config['NUM_TASK']
        # agglo_acc_list = []
        # agglo_ari_list = []
        # agglo_nmi_list = []

        # for t_id in range(total_num_task):
        #     acc = agglomerative_result_dict.get(f'Task_{t_id}_Accuracy', -1)
        #     ari = agglomerative_result_dict.get(f'Task_{t_id}_ARI', -1)
        #     nmi = agglomerative_result_dict.get(f'Task_{t_id}_NMI', -1)
        #     agglo_acc_list.append(acc)
        #     agglo_ari_list.append(ari)
        #     agglo_nmi_list.append(nmi)

        # # 결과를 result_dict에 저장
        # result_dict['Agglomerative_Acc_List'] = agglo_acc_list
        # result_dict['Agglomerative_ARI_List'] = agglo_ari_list
        # result_dict['Agglomerative_NMI_List'] = agglo_nmi_list

        # # 평균 계산
        # def compute_mean(values):
        #     valid_values = [v for v in values if v != -1]
        #     return np.round(np.mean(valid_values), 4) if valid_values else -1

        # agglo_log_dict = {
        #     'Agglomerative_Acc_Task_Seen': compute_mean(agglo_acc_list[:task_id + 1]),
        #     'Agglomerative_Acc_Task_All': compute_mean(agglo_acc_list),
        #     'Agglomerative_ARI_Task_Seen': compute_mean(agglo_ari_list[:task_id + 1]),
        #     'Agglomerative_ARI_Task_All': compute_mean(agglo_ari_list),
        #     'Agglomerative_NMI_Task_Seen': compute_mean(agglo_nmi_list[:task_id + 1]),
        #     'Agglomerative_NMI_Task_All': compute_mean(agglo_nmi_list),
        # }

        # # CIL 모드의 경우 전체 결과 추가
        # if self.params.il_mode == 'CIL':
        #     overall_acc = agglomerative_result_dict.get('Overall_Accuracy', -1)
        #     overall_ari = agglomerative_result_dict.get('Overall_ARI', -1)
        #     overall_nmi = agglomerative_result_dict.get('Overall_NMI', -1)
        #     agglo_log_dict['Agglomerative_Overall_Accuracy'] = overall_acc
        #     agglo_log_dict['Agglomerative_Overall_ARI'] = overall_ari
        #     agglo_log_dict['Agglomerative_Overall_NMI'] = overall_nmi

        # # 결과 로그 및 업데이트
        # for t_id in range(total_num_task):
        #     acc = agglo_acc_list[t_id]
        #     ari = agglo_ari_list[t_id]
        #     nmi = agglo_nmi_list[t_id]
        #     if acc != -1:
        #         logger.info(f'Updating Agglomerative Clustering result: task {task_id}, eval_task_id {t_id}, with Accuracy {acc}, ARI {ari}, NMI {nmi}')
        #         self.agglomerative_result_summary.update(task_id, t_id, Accuracy=acc, ARI=ari, NMI=nmi)

        # # 결과 요약 출력
        # if self.accelerator.is_main_process:
        #     logger.info(f'Result Summary Agglomerative Clustering After Task {task_id} =\n{self.agglomerative_result_summary.print_format()}')
        #     logger.info('Agglomerative Clustering Result = %s' % (agglo_log_dict))

        # self.accelerator.log(agglo_log_dict, step=self.global_step)

        
        # # 딥 클러스터링 평가
        # deep_clustering_result_dict = evaluate_deep_clustering_on_all_task(
        #     params=self.params,
        #     task_id=task_id,
        #     CL_dataset=self.CL_dataset,
        #     test_loader_list=self.test_loader_list,
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     accelerator=self.accelerator
        # )

        # # 결과 처리
        # total_num_task = self.CL_dataset.continual_config['NUM_TASK']
        # deep_acc_list = []
        # deep_ari_list = []
        # deep_nmi_list = []

        # for t_id in range(total_num_task):
        #     acc = deep_clustering_result_dict.get(f'Task_{t_id}_Accuracy', -1)
        #     ari = deep_clustering_result_dict.get(f'Task_{t_id}_ARI', -1)
        #     nmi = deep_clustering_result_dict.get(f'Task_{t_id}_NMI', -1)
        #     deep_acc_list.append(acc)
        #     deep_ari_list.append(ari)
        #     deep_nmi_list.append(nmi)

        # # 결과를 result_dict에 저장
        # result_dict['DeepClustering_Acc_List'] = deep_acc_list
        # result_dict['DeepClustering_ARI_List'] = deep_ari_list
        # result_dict['DeepClustering_NMI_List'] = deep_nmi_list

        # # 평균 계산
        # def compute_mean(values):
        #     valid_values = [v for v in values if v != -1]
        #     return np.round(np.mean(valid_values), 4) if valid_values else -1

        # deep_log_dict = {
        #     'DeepClustering_Acc_Task_Seen': compute_mean(deep_acc_list[:task_id + 1]),
        #     'DeepClustering_Acc_Task_All': compute_mean(deep_acc_list),
        #     'DeepClustering_ARI_Task_Seen': compute_mean(deep_ari_list[:task_id + 1]),
        #     'DeepClustering_ARI_Task_All': compute_mean(deep_ari_list),
        #     'DeepClustering_NMI_Task_Seen': compute_mean(deep_nmi_list[:task_id + 1]),
        #     'DeepClustering_NMI_Task_All': compute_mean(deep_nmi_list),
        # }

        # # CIL 모드의 경우 전체 결과 추가
        # if self.params.il_mode == 'CIL':
        #     overall_acc = deep_clustering_result_dict.get('Overall_Accuracy', -1)
        #     overall_ari = deep_clustering_result_dict.get('Overall_ARI', -1)
        #     overall_nmi = deep_clustering_result_dict.get('Overall_NMI', -1)
        #     deep_log_dict['DeepClustering_Overall_Accuracy'] = overall_acc
        #     deep_log_dict['DeepClustering_Overall_ARI'] = overall_ari
        #     deep_log_dict['DeepClustering_Overall_NMI'] = overall_nmi

        # # 결과 로그 및 업데이트
        # for t_id in range(total_num_task):
        #     acc = deep_acc_list[t_id]
        #     ari = deep_ari_list[t_id]
        #     nmi = deep_nmi_list[t_id]
        #     if acc != -1:
        #         logger.info(f'Updating Deep Clustering result: task {task_id}, eval_task_id {t_id}, with Accuracy {acc}, ARI {ari}, NMI {nmi}')
        #         self.deep_clustering_result_summary.update(task_id, t_id, Accuracy=acc, ARI=ari, NMI=nmi)

        # # 결과 요약 출력
        # if self.accelerator.is_main_process:
        #     logger.info(f'Result Summary Deep Clustering After Task {task_id} =\n{self.deep_clustering_result_summary.print_format()}')
        #     logger.info('Deep Clustering Result = %s' % (deep_log_dict))

        # self.accelerator.log(deep_log_dict, step=self.global_step)



        # Probing 부분: 모든 태스크에 대한 프로빙 성능 평가 및 LinearProb 결과 로그
        if self.params.is_probing:
            total_num_task = len(self.train_loader_list)

            if self.params.il_mode == 'CIL':
                # CIL 모드에서는 전체 태스크의 데이터 로더를 사용
                probing_train_loader_list = self.train_loader_list
                probing_test_loader_list = self.test_loader_list
                num_task = total_num_task  # num_task를 전체 태스크 수로 설정
            else:
                # TIL 모드에서는 현재 태스크와 다음 태스크까지의 데이터 로더를 사용
                num_task = task_id + 2  # 현재 태스크와 다음 태스크까지 포함
                num_task = min(num_task, total_num_task)  # 전체 태스크 수를 넘지 않도록 조정

                # 다음 태스크까지의 데이터 로더 준비
                probing_train_loader_list = self.train_loader_list[:num_task]
                probing_test_loader_list = self.test_loader_list[:num_task]

            prob_acc_dict = probing_on_all_task(
                params=self.params,
                task_id=task_id,
                CL_dataset=self.CL_dataset,
                train_loader_list=probing_train_loader_list,
                test_loader_list=probing_test_loader_list,
                model=self.model,
                tokenizer=self.tokenizer,
                accelerator=self.accelerator
            )

            # LinearProb 프로빙 결과만 로그 및 업데이트
            if 'LinearProb' in prob_acc_dict:
                prob_acc_list = prob_acc_dict['LinearProb']
                result_dict['LinearProb_List'] = prob_acc_list
                prob_log_dict = {
                    'LinearProb_Acc_Task_Seen': np.round(np.mean(prob_acc_list[:task_id + 1]), 3),
                    'LinearProb_Acc_Task_All': np.round(np.mean(prob_acc_list[:total_num_task]), 3)
                }

                # 각 태스크의 LinearProb 결과를 업데이트
                for t_id in range(total_num_task):
                    acc = prob_acc_list[t_id]
                    if acc != -1:
                        logger.info(f'Updating probing result for LinearProb: task {task_id}, eval_task_id {t_id}, with value {acc}')
                        self.probing_result_summary.update(task_id, t_id, acc)

                # 프로빙 결과 요약 출력
                if self.accelerator.is_main_process:
                    logger.info(f'Result Summary Probing After Task {task_id} (LinearProb) =\n{self.probing_result_summary.print_format()}')
                    logger.info('Probing Result (LinearProb) = %s' % (prob_log_dict))

                self.accelerator.log(prob_log_dict, step=self.global_step)

        return result_dict

    
    # def evaluate_model(self, task_id: int) -> dict:
    #     '''
    #     Evaluate the model and log the result

    #     Args:
    #         - task_id: task_id records how many tasks the model have learned previously

    #     Return:
    #         - result_dict: dictionary containing Test_Acc_List and optionally Next_Acc_List
    #     '''
    #     result_dict = {}
    #     log_dict = {}

    #     cur_task_id = task_id
    #     il_mode = self.params.il_mode

    #     # Evaluate current task and next task (if applicable)
    #     acc_list, acc_next_list = self.evaluate_all_seen_task_tc(cur_task_id, 'test', il_mode)
    #     result_dict['Test_Acc_List'] = acc_list

    #     # Log current task accuracy
    #     for t_id in range(cur_task_id + 1):
    #         log_dict['Test_Acc_Task_%d' % (t_id)] = acc_list[t_id]

    #     # Update result summary for current task
    #     logger.info(f'Updating task {cur_task_id}, eval_task_id {cur_task_id}, with value {acc_list[cur_task_id]}')
    #     self.result_summary.update(cur_task_id, cur_task_id, acc_list[cur_task_id])  # Update the task itself

    #     # Check if acc_next_list exists and is not None
    #     if acc_next_list is not None and cur_task_id + 1 < len(self.result_summary.result_summary):
    #         # Check if acc_next_list[cur_task_id] is valid before updating
    #         if acc_next_list[cur_task_id] is not None:
    #             logger.info(f'Updating next task: task {cur_task_id}, eval_task_id {cur_task_id+1}, with value {acc_next_list[cur_task_id]}')
    #             self.result_summary.update(cur_task_id, cur_task_id + 1, acc_next_list[cur_task_id])  # Update the next task

    #     # Log results
    #     log_dict['Test_Acc_Task_Seen'] = np.round(np.mean(acc_list[:cur_task_id + 1]), 3)
    #     if self.params.classifier == 'None':
    #         log_dict['Test_Acc_Task_All'] = np.round(np.mean(acc_list), 3)

    #     if self.accelerator.is_main_process:
    #         logger.info('Mode = %s, Test Result = %s' % (il_mode, log_dict))
    #     self.accelerator.log(log_dict, step=self.global_step)
        
    #     if self.params.is_probing:
    #         prob_acc_dict = probing_on_all_task(
    #             params=self.params,
    #             task_id=task_id,
    #             CL_dataset=self.CL_dataset, 
    #             train_loader_list=self.train_loader_list, 
    #             test_loader_list=self.test_loader_list, 
    #             model=self.model, 
    #             tokenizer=self.tokenizer, 
    #             accelerator=self.accelerator
    #         )
    #         prob_log_dict = {}
    #         for prob_metric in prob_acc_dict.keys():
    #             result_dict['%s_List'%(prob_metric)] = prob_acc_dict[prob_metric]
    #             prob_log_dict['%s_Acc_Task_Seen'%(prob_metric)] = np.round(np.mean(prob_acc_dict[prob_metric][:cur_task_id+1]),3)
    #             prob_log_dict['%s_Acc_Task_All'%(prob_metric)] = np.round(np.mean(prob_acc_dict[prob_metric]),3)
    #         if self.accelerator.is_main_process:
    #             logger.info('Probing Result = %s'%(prob_log_dict))
    #         self.accelerator.log(prob_log_dict, step=self.global_step)

    #     return result_dict

    # def evaluate_all_seen_task_tc(self, cur_task_id: int, phase: str, il_mode: str) -> tuple:
    #     '''
    #     Evaluate the model on all seen tasks, including current and next tasks

    #     Params:
    #         - cur_task_id: the ID of the current task
    #         - phase: 'train', 'dev', or 'test'
    #         - il_mode: 'CIL' or 'TIL'

    #     Return:
    #         - acc_list: list of accuracies for all tasks evaluated
    #         - acc_next_task_list: list of accuracies for the next task (if exists)
    #     '''
    #     assert phase in ['train', 'test', 'dev']

    #     acc_list = []
    #     acc_next_task_list = None

    #     # Evaluate on all seen tasks
    #     if self.params.classification_type == 'sentence-level':
    #         save_dict_all = None

    #         for eval_t_id in range(cur_task_id + 1):
    #             # Evaluate current task
    #             acc, acc_next_task = self.evaluate_current_task(eval_t_id, cur_task_id, phase, il_mode)
    #             acc_list.append(acc)

    #             # If next task exists, store its result
    #             if acc_next_task is not None:
    #                 if acc_next_task_list is None:
    #                     acc_next_task_list = []
    #                 acc_next_task_list.append(acc_next_task)

    #         save_dict_all = [save_dict_all]  # transform to list for gather_object() to collect correctly
    #         gathered_save_dict_all = gather_object(save_dict_all)

    #         if self.accelerator.is_main_process:
    #             if gathered_save_dict_all is not None:
    #                 with open(os.path.join(self.params.dump_path, '%s_cur_task_%d_save_result.npy' % (phase, cur_task_id)), 'wb') as f:
    #                     np.save(f, gathered_save_dict_all)

    #     return acc_list, acc_next_task_list
    
    def evaluate_all_seen_task_tc(self, cur_task_id: int, phase: str, il_mode: str) -> tuple:
        '''
        Evaluate the model on all seen tasks, including current and next tasks

        Params:
            - cur_task_id: the ID of the current task
            - phase: 'train', 'dev', or 'test'
            - il_mode: 'CIL' or 'TIL'

        Return:
            - acc_list: list of accuracies for all tasks evaluated
            - acc_next_task_list: list of accuracies for the next task (if exists)
        '''
        assert phase in ['train', 'test', 'dev']

        acc_list = []
        acc_next_task_list = None

        # Evaluate on all seen tasks
        save_dict_all = None

        for eval_t_id in range(cur_task_id + 1):
            # Evaluate current task
            if self.params.classification_type == 'sentence-level':
                acc, acc_next_task = self.evaluate_current_task(eval_t_id, cur_task_id, phase, il_mode)
            elif self.params.classification_type == 'word-level':
                acc, acc_next_task = self.evaluate_current_task(eval_t_id, cur_task_id, phase, il_mode)
            else:
                raise ValueError(f"Unsupported classification type: {self.params.classification_type}")

            acc_list.append(acc)

            # If next task exists, store its result
            if acc_next_task is not None:
                if acc_next_task_list is None:
                    acc_next_task_list = []
                acc_next_task_list.append(acc_next_task)

        save_dict_all = [save_dict_all]  # transform to list for gather_object() to collect correctly
        gathered_save_dict_all = gather_object(save_dict_all)

        if self.accelerator.is_main_process:
            if gathered_save_dict_all is not None:
                with open(os.path.join(self.params.dump_path, f'{phase}_cur_task_{cur_task_id}_save_result.npy'), 'wb') as f:
                    np.save(f, gathered_save_dict_all)

        return acc_list, acc_next_task_list

    @abstractmethod
    def evaluate_current_task(self, eval_task_id: int, cur_task_id: int, phase: str, il_mode: str) -> tuple:
        '''
        Evaluate the model on the current task

        Params:
            - eval_task_id: the ID of the task to be evaluated
            - cur_task_id: the ID of the current task
            - phase: 'train', 'dev', or 'test'
            - il_mode: 'CIL' or 'TIL'

        Return:
            - acc: accuracy for the evaluated task
            - acc_next_task: accuracy for the next task (if exists), otherwise None
        '''
        pass
    # ===========================================================================================


    # ================================= Epoch-Level Functions ====================================
    @abstractmethod
    def train_epochs(self, task_id):
        pass
    # ===========================================================================================
    
    def finish_training(self):
        '''
            Finish training: print the result
        '''
        log_dict = {}
        il_mode = self.params.il_mode
        num_tasks = self.CL_dataset.continual_config['NUM_TASK']
        if self.accelerator.is_main_process:
            logger.info('Mode = %s, Summary Test Acc = \n%s' % (il_mode, self.result_summary.print_format()))
            result_dict = self.result_summary.get_value()
            result_matrix = result_dict  # 이미 numpy 배열이라고 가정
            # 메트릭 계산
            bwt_acc = compute_backward_transfer(result_matrix)
            fwt_acc = compute_forward_transfer(result_matrix)
            fgt_acc = compute_forgetting(result_matrix)
            aver_acc = compute_average_acc(result_matrix)
            aver_inc_acc = compute_average_inc_acc(result_matrix)
            log_dict['Test_Aver_ACC'] = aver_acc
            log_dict['Test_Bwt_ACC'] = bwt_acc
            log_dict['Test_Fwt_ACC'] = fwt_acc
            log_dict['Test_Fgt_ACC'] = fgt_acc
            log_dict['Test_Aver_Inc_ACC'] = aver_inc_acc
            logger.info('Mode = %s, Summary Result = \n%s'%(il_mode,log_dict))    

        # 클러스터링 결과 처리 추가
        clustering_methods = ['kmeans', 'gmm', 'spectral', 'agglomerative', 'deep_clustering']
        metrics_list = ['Accuracy', 'ARI', 'NMI']
        for method in clustering_methods:
            result_summary_attr = f'{method}_result_summary'
            if hasattr(self, result_summary_attr):
                result_summary = getattr(self, result_summary_attr)
                if self.accelerator.is_main_process:
                    logger.info(f'Mode = {il_mode}, Summary {method.capitalize()} Results:')
                    # 각 메트릭별로 결과 매트릭스 가져오기
                    for metric in metrics_list:
                        result_matrix = result_summary.get_metric_matrix(metric)
                        if result_matrix is None:
                            continue  # 해당 메트릭에 대한 결과가 없으면 넘어감

                        # 메트릭 계산
                        method_bwt = compute_backward_transfer(result_matrix)
                        method_fwt = compute_forward_transfer(result_matrix)
                        method_fgt = compute_forgetting(result_matrix)
                        method_aver = compute_average_acc(result_matrix)
                        method_aver_inc = compute_average_inc_acc(result_matrix)

                        # 로그 딕셔너리에 저장
                        method_log_dict = {}
                        method_log_dict[f'{method.capitalize()}_{metric}_Aver'] = method_aver
                        method_log_dict[f'{method.capitalize()}_{metric}_Bwt'] = method_bwt
                        method_log_dict[f'{method.capitalize()}_{metric}_Fwt'] = method_fwt
                        method_log_dict[f'{method.capitalize()}_{metric}_Fgt'] = method_fgt
                        method_log_dict[f'{method.capitalize()}_{metric}_Aver_Inc'] = method_aver_inc

                        logger.info(f'{method.capitalize()} {metric} Results: {method_log_dict}')
                        self.accelerator.log(method_log_dict, step=self.global_step)
                        print(f'{method.capitalize()} {metric} Result Matrix:')
                        print(result_matrix)

        # 기존의 probing_result_summary에 대한 처리
        if hasattr(self, 'probing_result_summary'):
            probing_log_dict = {}
            if self.accelerator.is_main_process:
                logger.info('Mode = %s, Summary Probing Acc = \n%s' % (il_mode, self.probing_result_summary.print_format()))
            # Compute Forward and Backward Transfer according to Probing Result Summary for the whole learning process
            probing_bwt_acc = compute_backward_transfer(self.probing_result_summary.get_value())
            probing_fwt_acc = compute_forward_transfer(self.probing_result_summary.get_value()) 
            probing_fgt_acc = compute_forgetting(self.probing_result_summary.get_value()) 
            probing_aver_acc = compute_average_acc(self.probing_result_summary.get_value()) 
            probing_aver_inc_acc = compute_average_inc_acc(self.probing_result_summary.get_value()) 
            probing_log_dict['Probing_Aver_ACC'] = probing_aver_acc
            probing_log_dict['Probing_Bwt_ACC'] = probing_bwt_acc
            probing_log_dict['Probing_Fwt_ACC'] = probing_fwt_acc
            probing_log_dict['Probing_Fgt_ACC'] = probing_fgt_acc
            probing_log_dict['Probing_Aver_Inc_ACC'] = probing_aver_inc_acc

            if self.accelerator.is_main_process:
                logger.info('Mode = %s, Summary Probing Result = \n%s' % (il_mode, probing_log_dict))    
            self.accelerator.log(probing_log_dict, step=self.global_step)
            print(self.probing_result_summary.get_value())
        
        # Delete checkpoints
        # if not self.params.save_ckpt:
        #     for file_name in os.listdir(self.params.dump_path):
        #         if file_name[-4:] == '.pth':
        #             os.remove(os.path.join(self.params.dump_path,file_name))

        # End Wandb
        self.accelerator.end_training()

