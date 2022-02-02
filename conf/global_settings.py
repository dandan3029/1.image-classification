""" configurations for this project

author baiyu
"""
import os
# 导入os，实际上没有用到，可以考虑删掉
from datetime import datetime
# 导入datetime，方便获取当前时间

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# 定义CIFAR100训练集三通道的均值，为了预处理数据
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
# 定义CIFAR100训练集三通道的方差，为了预处理数据

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'
# 保存模型参数文件的路径

#total training epoches
EPOCH = 200
# 定义了批次的数量
MILESTONES = [60, 120, 160]
# 定义了训练过程中重要的批次节点

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
# 获取当前时间，并将其格式化

#tensorboard log dir
LOG_DIR = 'runs'
# tensorboard的log目录

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
# 每1个批次保存一次模型参数







