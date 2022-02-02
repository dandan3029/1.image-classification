""" train and test dataset

author baiyu
"""
import os
# 导入os，便于处理路径
import sys
# 导入sys，实际上没有用到
import pickle
# 导入pickle，用于加载数据

from skimage import io
# 导入skimage.io，用于图像的读写可视化等，实际上没有用到
import matplotlib.pyplot as plt
# 导入matplotlib.pyplot，用于绘图，没有用到
import numpy
# 导入numpy，数学计算
import torch
# 导入torch框架，用于gpu并行计算
from torch.utils.data import Dataset
# 导入torch自带的Dataset，作为自定义的dataset的父类

class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """
    # 定义CIFAR100Train数据集类，用于加载并处理数据，使其可以灌入模型中

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        # 定义初始化函数
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            # 打开数据集所在文件夹
            self.data = pickle.load(cifar100, encoding='bytes')
            # 加载文件
        self.transform = transform
        # 为类的transform赋值

    def __len__(self):
        # 定义返回数据长度的函数
        return len(self.data['fine_labels'.encode()])
        # 返回数据长度

    def __getitem__(self, index):
        # 定义获取一条数据的函数
        label = self.data['fine_labels'.encode()][index]
        # 数据标签
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        # r通道
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        # g通道
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        # b通道
        image = numpy.dstack((r, g, b))
        # 定义image的numpy数组

        if self.transform:
            # 如果transform不为空
            image = self.transform(image)
            # 对图像进行变换
        return label, image
        # 返回标签和图像

class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """
    # 定义CIFAR100Test数据集类，用于加载并处理数据，使其可以灌入模型中

    def __init__(self, path, transform=None):
        # 定义初始化函数
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            # 打开数据集所在文件夹
            self.data = pickle.load(cifar100, encoding='bytes')
            # 加载文件
        self.transform = transform
        # 为类的transform赋值

    def __len__(self):
        # 定义返回数据长度的函数
        return len(self.data['data'.encode()])
        # 返回数据长度

    def __getitem__(self, index):
        # 定义获取一条数据的函数
        label = self.data['fine_labels'.encode()][index]
        # 数据标签
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        # r通道
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        # g通道
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        # b通道
        image = numpy.dstack((r, g, b))
        # 定义image的numpy数组

        if self.transform:
            # 如果transform不为空
            image = self.transform(image)
            # 对图像进行变换
        return label, image
        # 返回标签和图像

