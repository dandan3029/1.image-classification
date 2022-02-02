import torch,argparse
from utils import get_network

parser = argparse.ArgumentParser()
# 定义一个ArgumentParser对象
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
# 添加命令行参数-gpu，默认值为False，可缺省
args = parser.parse_args()

# googlenet
googlenet_pretrained_path = 'pretrained/googlenet-1378be20.pth'
googlenet_trained_path = 'checkpoint/googlenet/googlenet-12-regular.pth'

# resnet18
resnet18_pretrained_path = 'pretrained/resnet18-5c106cde.pth'

# mobilenet
mobilenet_pretrained_path = 'pretrained/mobilenet_v2-b0353104.pth'

googlenet_pretrained_weight = torch.load(googlenet_pretrained_path)
googlenet_trained_weight = torch.load(googlenet_trained_path)

resnet18_pretrained_weight = torch.load(resnet18_pretrained_path)
resnet18 = get_network(args)

# mobilenet_pretrained_weight = torch.load(mobilenet_pretrained_path)
# mobilenet = get_network(args)

print("googlenet_pretrained_weight:")
print(googlenet_pretrained_weight.keys())
print("googlenet_trained_weight:")
print(googlenet_trained_weight.keys())

print("resnet18 pretrained weight:")
print(resnet18_pretrained_weight.keys())
print("resnet18 weight:")
print(resnet18.state_dict().keys())

# print("mobilenet pretrained weight:")
# print(mobilenet_pretrained_weight)
# print("mobilenet weight:")
# print(mobilenet.state_dict())