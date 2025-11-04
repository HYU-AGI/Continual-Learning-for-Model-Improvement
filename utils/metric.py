from typing import Any
import numpy as np 
import logging

logger = logging.getLogger()

# class ResultSummary(object):

#     # ============================================================================
#     # Result Summary Matrix
#     #                           Task 0   Task 1   Task 2
#     #                        -----------------------------
#     #  Learning After Task 0 |        |         |        |
#     #                        -----------------------------
#     #  Learning After Task 1 |        |         |        |
#     #                        -----------------------------
#     #  Learning After Task 2 |        |         |        |
#     #                        -----------------------------
#     # ============================================================================

#     def __init__(self, num_task: int) -> None:
#         self.result_summary = np.ones((num_task,num_task))*-1

#     def update(self, task_id: int, eval_task_id: int, value: float) -> None:
#         '''
#             Update the metric:

#             self.result_summary[task_id,eval_task_id] = value

#             Args:
#                 - task_id: evaluation after learning the {task_id}-th task
#                 - eval_task_id: the evaluation result of the {task_id}-th task
#                 - value: the value to be updated
#         '''
#         self.result_summary[task_id,eval_task_id] = value

#     def get_value(self) -> np.array:

#         return self.result_summary

#     def print_format(self, round: int=2) -> np.array:

#         return np.around(self.result_summary,round)

class ResultSummary(object):

    def __init__(self, num_task: int) -> None:
        self.result_summary = np.ones((num_task, num_task)) * -1  # Initialize the summary matrix with -1

    def update(self, task_id: int, eval_task_id: int, value: float) -> None:
        '''
            Update the metric for the current task and next task:

            self.result_summary[task_id, eval_task_id] = value

            Args:
                - task_id: evaluation after learning the {task_id}-th task
                - eval_task_id: the evaluation result of the {task_id}-th task
                - value: the value to be updated
        '''
        logger.info(f"ResultSummary update: task_id={task_id}, eval_task_id={eval_task_id}, value={value}")
        # Update the evaluation result in the correct position
        self.result_summary[task_id, eval_task_id] = value

    def get_value(self) -> np.array:
        ''' Return the current result summary matrix. '''
        return self.result_summary

    def print_format(self, round: int = 2) -> np.array:
        ''' Return the formatted result summary matrix, rounded to the given decimal points. '''
        return np.around(self.result_summary, round)
    
class ResultSummary2(object):

    def __init__(self, num_task):
        self.num_task = num_task
        self.metrics = {}  # 메트릭별로 결과 저장

    def update(self, train_task_id, eval_task_id, **metrics):
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics:
                # 초기화
                self.metrics[metric_name] = np.full((self.num_task, self.num_task), np.nan)
            self.metrics[metric_name][train_task_id, eval_task_id] = value

    def get_metric_matrix(self, metric_name):
        return self.metrics.get(metric_name, None)

    def print_format(self):
        output = ''
        for metric_name, matrix in self.metrics.items():
            output += f'\nMetric: {metric_name}\n{matrix}\n'
        return output

    def get_value(self):
        # 필요에 따라 수정
        return self.metrics