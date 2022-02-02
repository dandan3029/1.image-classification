# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
# 用于解析命令行参数
import time
# 用于获取当前时间，计算模型训练的时间
from datetime import datetime
# 用于获取当前时间，这里没有用到这种方法

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
# 导入conf文件夹下的settings对象
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
# 导入utils文件夹下的get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

def train(epoch):
    # 训练模型
    start = time.time()
    # 记录训练开始的时间
    net.train()
    # model.train() 让model变成训练模式，此时 dropout和batch normalization的操作在训练中起到防止网络过拟合的作用
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        # 分批次遍历数据集中的数据
        if epoch <= args.warm:
            # 判断epoch是否小于warmup的值
            warmup_scheduler.step()
            # 先进行lr较小的训练，防止lr过大梯度爆炸

        if args.gpu:
            # 判断是否使用gpu
            labels = labels.cuda()
            # 将labels转换为cuda张量
            images = images.cuda()
            # 将images转换为cuda张量

        optimizer.zero_grad()
        # 将梯度初始化为零
        outputs = net(images)
        # 将images输入网络中得到输出
        loss = loss_function(outputs, labels)
        # 计算loss
        loss.backward()
        # 反向传播
        optimizer.step()
        # 更新梯度

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))
        # 输出训练过程中的指标等 loss，lr



    finish = time.time()
    # 记录训练完成的时间

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    # 输出训练过程所使用的时间


@torch.no_grad()
# 该注解表示不需要计算梯度，也不会进行反向传播
def eval_training(epoch):
    # 定义验证训练效果的函数

    start = time.time()
    # 记录验证开始的时间
    net.eval()
    # model.eval()，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。

    test_loss = 0.0 # cost function error
    # 将test_loss初始化为0
    correct = 0.0
    # 将correct初始化为0

    for (images, labels) in cifar100_test_loader:
        # 从test_loader中获取数据
        if args.gpu:
            # 判断是否使用gpu
            images = images.cuda()
            # 将images转换为cuda张量
            labels = labels.cuda()
            # 将labels转换为cuda张量

        outputs = net(images)
        # 将images输入网络得到输出
        loss = loss_function(outputs, labels)
        # 计算loss
        test_loss += loss.item()
        # 累加test_loss
        _, preds = outputs.max(1)
        # 获取预测结果
        correct += preds.eq(labels).sum()
        # 预测正确样本的累加

    finish = time.time()
    # 记录验证结束的时间
    if args.gpu:
        # 判断是否使用gpu
        print('Use GPU')
        # 输出提示
    print('Evaluating Network.....')
    # 输出提示
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    # 输出验证指标
    print()
    # 输出换行
    return correct.float() / len(cifar100_test_loader.dataset)
    # 返回测试集上的acc


if __name__ == '__main__':
    # 判断是否为程序的入口
    # 当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
    # 当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。

    parser = argparse.ArgumentParser()
    # 定义一个ArgumentParser对象
    parser.add_argument('-net', type=str, required=True, help='net type')
    # 添加命令行参数-net，必须要有，类型为str
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    # 添加命令行参数-gpu，默认值为False，可缺省
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    # 添加命令行参数-b，默认值为128，可缺省
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    # 添加命令行参数-warm，默认值为1，可缺省
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    # 添加命令行参数-lr，默认值为0.1，可缺省
    parser.add_argument('-resume', default=None, type=int, help='Checkpoint state_dict file to resume training from')
    # 添加命令行参数-resume，默认值为None，不使用断点续训
    args = parser.parse_args()
    # 解析命令行参数

    net = get_network(args)
    # 根据-net获取相应的模型

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=args.b,
        shuffle=True
    )
    # 数据预处理，进行归一化处理，随机打乱顺序
    # 4个进程加载数据

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=args.b,
        shuffle=True
    )
    # 数据预处理，进行归一化处理，随机打乱顺序
    # 4个进程加载数据

    loss_function = nn.CrossEntropyLoss()
    # 采用交叉熵损失
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # 采用带动量的随机梯度下降优化器
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    # 基于epoch训练次数进行学习率调整
    # 当训练epoch达到milestones值时,初始学习率乘以gamma得到新的学习率;
    iter_per_epoch = len(cifar100_training_loader)
    # 每个epoch有training_loader长度次迭代
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    # 先用小lr进行warmup，防止梯度爆炸
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net)
    # 定义模型参数文件的路径 checkpoint/args.net/time

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        # 判断checkpoint_path是否存在
        os.makedirs(checkpoint_path)
        # 如果路径不存在则创建该路径
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    # 更新checkpoint_path

    best_acc = 0.0
    # 初始化best_acc为0
    if args.resume:
        state_dict = torch.load(checkpoint_path.format(net=args.net, epoch=args.resume, type='regular'))
        net.load_state_dict(state_dict)
        print("Load epoch {} state dict successfully...".format(args.resume))
        best_acc = eval_training(args.resume)
        for epoch in range(args.resume+1, settings.EPOCH):
            # 循环EPOCH次
            if epoch > args.warm:
                # 判断epoch是否大于warm
                train_scheduler.step(epoch)
                # 如果epoch大于warm，则开始按照该方法进行梯度更新

            train(epoch)
            # 调用train函数
            acc = eval_training(epoch)
            # 在test上计算acc

            # start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                # 如果有了更好的结果
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                # 保存模型参数
                best_acc = acc
                # 保存模型最好的acc
                continue
                # 结束该次循环

            if not epoch % settings.SAVE_EPOCH:
                # 每个epoch保存一次模型参数
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
                # 保存模型参数
                if best_acc < acc:
                    best_acc = acc
                    print("The {} epoch model become better!".format(epoch))
                # 保存模型最好的acc

    else:
        for epoch in range(1, settings.EPOCH):
            # 循环EPOCH次
            if epoch > args.warm:
                # 判断epoch是否大于warm
                train_scheduler.step(epoch)
                # 如果epoch大于warm，则开始按照该方法进行梯度更新

            train(epoch)
            # 调用train函数
            acc = eval_training(epoch)
            # 在test上计算acc

            #start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                # 如果有了更好的结果
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                # 保存模型参数
                best_acc = acc
                # 保存模型最好的acc
                continue
                # 结束该次循环

            if not epoch % settings.SAVE_EPOCH:
                # 每10个epoch保存一次模型参数
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
                # 保存模型参数

