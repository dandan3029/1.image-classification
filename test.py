#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
# 用于解析命令行参数

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
# 导入conf文件夹下的settings对象
from utils import get_network, get_test_dataloader
# 导入utils文件夹下的get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
from torch.hub import load_state_dict_from_url
# 用于从url加载模型参数

if __name__ == '__main__':
    # 判断是否为程序的入口
    # 当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
    # 当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。

    parser = argparse.ArgumentParser()
    # 定义一个ArgumentParser对象
    parser.add_argument('-net', type=str, required=True, help='net type')
    # 添加命令行参数-net，必须要有，类型为str
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    # 添加命令行参数-weights，必须要有，类型为str
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    # 添加命令行参数-gpu，默认值为False，可缺省
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    # 添加命令行参数-b，默认值为128，可缺省
    args = parser.parse_args()
    # 解析命令行参数

    net = get_network(args)
    # 根据-net获取相应的模型

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=args.b,
        shuffle=False
    )
    # 数据预处理，进行归一化处理，随机打乱顺序
    # 1个进程加载数据
    net.load_state_dict(torch.load(args.weights))
    # 加载模型参数
    # print(net)
    # 输出模型结构
    net.eval()
    # model.eval()，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。

    correct_1 = 0.0
    # top1 正确的个数
    correct_5 = 0.0
    # top5 正确的个数
    total = 0
    # 没有用的变量
    results = []
    with torch.no_grad():
        # 不计算梯度
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            # 遍历test_loader
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                # 判断是否使用gpu
                image = image.cuda()
                # 将images转换为cuda张量
                label = label.cuda()
                # 将labels转换为cuda张量

            output = net(image)
            # 将images输入网络得到输出
            # _, pred = output.topk(5, 1, largest=True, sorted=True)
            _, pred = output.topk(1, 1, largest=True, sorted=True)
            # 获取top5预测结果 shape为 batch_size * k
            preds = pred.cpu().numpy()[:,0].tolist()
            results.extend(preds)
    print(len(results))
    with open('results_{}.txt'.format(args.net), 'w') as f:
        for item in results:
            f.write(str(item)+'\n')