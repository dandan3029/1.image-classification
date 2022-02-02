
import argparse
# 用于解析命令行参数，这里没有用到
import glob
# glob模块提供了一个函数用于从目录通配符搜索中生成文件列表:
# >>> import glob
# >>> glob.glob('*.py')
# ['primes.py', 'random.py', 'quote.py']
import os
# 常用os.path，这里没有用到

import cv2
# 导入opencv，没用到
import torch
# 导入torch
import torch.nn as nn
# 导入nn，用于搭建神经网络，没用到
import torch.optim as optim
# 导入优化器
from torch.utils.data import DataLoader
# 导入dataloader
import numpy as np
# 导入numpy

from torchvision import transforms
# 导入transforms，用于对PIL.Image进行变换，这里没有用到
from conf import settings
# 导入settings对象，没有用到
from utils import *
# 导入utils

import matplotlib
# 导入matplotlib绘图
matplotlib.use('Agg')
# 选择用于呈现和GUI集成的后端
import matplotlib.pyplot as plt
# 导入plt用于绘图，没有用到


from torch.optim.lr_scheduler import _LRScheduler
# 导入_LRScheduler类


class FindLR(_LRScheduler):
    # 定义FindLR类
    """exponentially increasing learning rate

    Args:
        optimizer: optimzier(e.g. SGD)
        num_iter: totoal_iters
        max_lr: maximum  learning rate
    """
    def __init__(self, optimizer, max_lr=10, num_iter=100, last_epoch=-1):
        # 定义初始化函数，并提供max_lr，num_iter，last_epoch的缺省值/默认值

        self.total_iters = num_iter
        # 总迭代次数
        self.max_lr = max_lr
        # 最大学习率
        super().__init__(optimizer, last_epoch)
        # 调用父类_LRScheduler的初始化函数

    def get_lr(self):
        # 获取当前学习率

        return [base_lr * (self.max_lr / base_lr) ** (self.last_epoch / (self.total_iters + 1e-32)) for base_lr in self.base_lrs]
        # 返回当前学习率，学习率指数衰减