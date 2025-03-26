'''
@Project ：NLPNewsClassification 
@File    ：timer.py
@Author  ：DZY
@Date    ：2025/3/14 17:05 
'''
import time

import numpy as np


class Timer():
    def __init__(self,steps_time=None):
        if steps_time is None:
            self.steps_time = []
        else:
            self.steps_time = steps_time

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()
        self.steps_time.append(self.stop_time - self.start_time)
        return self.steps_time[-1]

    def get_time_diff(self):
        return self.steps_time[-1]

    def get_steps_time(self):
        return self.steps_time

    def get_step_time_diff(self, start_step_index, end_step_index):
        return sum(self.steps_time[start_step_index:end_step_index])

    def get_total_time(self):
        return sum(self.steps_time)

    def get_avg_time(self):
        return self.get_total_time() / len(self.steps_time)

    def get_cumulate_time(self):
        """

        Returns:
            返回累积时间

        """
        return np.array(self.steps_time).cumsum().tolist()


# 将训练时长格式化为 小时:分:秒
def format_duration(seconds):
    # 计算小时、分钟和秒
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    # 格式化为 hh:mm:ss
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# pretrain_timer=Timer()
# pretrain_info_interval=2
# total_mlm_loss=0.0
#
# for current_pretrain_iter_step in range(0,20):
#     pretrain_timer.start()
#     time.sleep(2)
#     pretrain_timer.stop()
#     total_mlm_loss+=random.random()*(current_pretrain_iter_step+1/100)
#
#     # print(pretrain_timer.get_time_diff())
#     # print(pretrain_timer.get_cumulate_time())
#     if (current_pretrain_iter_step + 1) % pretrain_info_interval == 0:
#         current_pretrain_iter_time = sum(pretrain_timer.steps_time[
#                                          current_pretrain_iter_step + 1 - pretrain_info_interval:current_pretrain_iter_step + 1])
#         cum_time_list = pretrain_timer.get_cumulate_time()
#         print(
#             f"Iter Steps: {current_pretrain_iter_step + 2 - pretrain_info_interval} - {current_pretrain_iter_step + 1}, "
#             f"Avg MLM Loss: {total_mlm_loss / (current_pretrain_iter_step + 1):.4f}, "
#             f"Iter Time: {current_pretrain_iter_time} sec, "
#             f"Cumulative Iter Time: {cum_time_list[current_pretrain_iter_step]} sec")
#
# print(pretrain_timer.get_total_time())
