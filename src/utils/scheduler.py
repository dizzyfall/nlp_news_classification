'''
@Project ：NLPNewsClassification 
@File    ：scheduler.py
@Author  ：DZY
@Date    ：2025/3/25 11:35 
'''
import numpy as np


class BERTScheduler():
    def __init__(self, optimizer, num_hiddens=768, warmup_steps=10000, current_step=0):
        """

        Args:
            optimizer: 优化器对象
            num_hiddens: 词向量长度
            warmup_steps: warmup步数
            current_step: 当前epoch（用于恢复训练）

        Returns:

        """
        self.optimizer = optimizer
        self.init_lr = np.power(num_hiddens, -0.5)
        self.warmup_steps = warmup_steps
        self.current_step = current_step
        self.step()

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # warmup阶段，线性增加学习率
            return self.init_lr * np.power(self.warmup_steps, -1.5) * self.current_step
        else:
            # 线性衰退
            return self.init_lr * np.power(self.current_step, -0.5)

    def step(self):
        """更新学习率"""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
