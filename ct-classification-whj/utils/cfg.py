# -*- coding:utf-8 -*-
import os
# import torchvision

##数据集的类别
NUM_CLASSES = 206

# 训练时batch的大小
BATCH_SIZE = 32

# 网络默认输入图像的大小
INPUT_SIZE = 300
# 训练最多的epoch
MAX_EPOCH = 100
# 使用gpu的数目
GPUS = 2
# 从第几个epoch开始resume训练，如果为0，从头开始
RESUME_EPOCH = 0

WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
# 初始学习率
LR = 1e-3

# 采用的模型名称
# model_name = 'resnext101_32x32d'
# model_name = 'ShuffleNetV2'
# model_name = 'resnet101'
# model_name = 'MobileNetV3'
# model_name = 'DenseNet'
model_name = 'resnet152'

# weights_path = 'weights/resnext101_32x32d/shuffle_10.pth'
# weights_path = 'checkpoint/expDes/resnet152_100.pth'
weights_path = './resnet152_100.pth'

# from imgmodels import Resnet50, Resnet101, Resnext101_32x8d, Resnext101_32x16d, Densenet121, Densenet169, Mobilenetv2, \
#     Efficientnet, Resnext101_32x32d, Resnext101_32x48d

# MODEL_NAMES = {
#     'resnext101_32x8d': Resnext101_32x8d,
#     'resnext101_32x16d': Resnext101_32x16d,
#     'resnext101_32x48d': Resnext101_32x48d,
#     'resnext101_32x32d': Resnext101_32x32d,
#     'resnet50': Resnet50,
#     'resnet101': Resnet101,
#     'densenet121': Densenet121,
#     'densenet169': Densenet169,
#     'moblienetv2': Mobilenetv2,
#     'efficientnet-b7': Efficientnet,
#     'efficientnet-b8': Efficientnet
# }

BASE = os.getcwd()

# 训练好模型的保存位置
SAVE_FOLDER = os.path.join(BASE, 'weights/')

# 数据集的存放位置
# TRAIN_LABEL_DIR =BASE + 'train.txt'
# VAL_LABEL_DIR = BASE + 'val.txt'
TEST_LABEL_DIR = os.path.join(BASE, './test.txt')

##训练完成，权重文件的保存路径,默认保存在trained_model下
TRAINED_MODEL = os.path.join(BASE, weights_path)
