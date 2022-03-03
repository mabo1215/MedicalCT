import argparse
import math
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import auc, roc_curve
from torchvision import models, transforms
from tqdm import tqdm

from utils import cfg, maketxt
from utils.utils import *


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


# def get_test_transform(mean=mean, std=std, size=0):
#     return transforms.Compose([
#         Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
#         transforms.CenterCrop(size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ])
def get_test_transform(mean, std, size=0):
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])


# def load_checkpoint(filepath,model_name):
#     print(filepath)
#     checkpoint = torch.load(filepath)
#     if model_name == "DenseNet":
#         model = models.DenseNet()
#     elif model_name == "Resnet152":
#         model = models.resnet152(pretrained=True)
#     elif model_name == "Resnet18":
#         model = models.resnet18(pretrained=True)
#     elif model_name == "Mobilenet_v3_small":
#         model = models.mobilenet_v3_small(pretrained=True, progress=True)
#     elif model_name == "SqueezeNet":
#         model = models.squeezenet1_1(pretrained=True)
#     elif model_name == "Vgg":
#         model = models.vgg11(pretrained=True)
#     elif model_name == "Googlenet":
#         model = models.googlenet(pretrained=True, progress=True)
#     elif model_name == "Shufflenetv2":
#         model = models.shufflenet_v2_x1_0(pretrained=True)
#     elif model_name == "Efficientnet":
#         model = models.efficientnet_b7(pretrained=True)
#     elif model_name == "Mnasnet1_0":
#         model = models.mnasnet1_0(pretrained=True, progress=True)
#     elif model_name == "Regnet":
#         model = models.regnet_y_800mf(pretrained=True, progress=True)
#     elif model_name == "Alexnet":
#         model = models.alexnet(pretrained=True)
#     elif model_name == "Inceptionv3":
#         model = models.inception_v3(pretrained=True)
#     else:
#         model = models.resnet101(pretrained=True)
#     # model = checkpoint['model']  # extra model
#     model.load_state_dict(checkpoint)  # load model
#     for parameter in model.parameters():
#         parameter.requires_grad = False
#     model.eval()
#     return model


def predict(model, model_name, test_percent):
    # 读入模型
    model = load_checkpoint(model, model_name)
    print('..... Finished loading model! ......')
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # 将模型放置在gpu上运行
    dev = 0
    if torch.cuda.is_available():
        dev = 1
    if dev == 1:
        model.cuda()
    pred_list, _id = [], []
    # test_percent = 0.1
    num = len(imgs)
    list = range(num)
    test = int(test_percent * num)
    testlist = random.sample(list, test)
    for i in tqdm(list):
        # img_path = imgs[i].strip()
        if i in testlist:
            img_path = imgs[i].strip()
            # print(img_path)
            _id.append(os.path.basename(img_path).split('.')[0])
            img = Image.open(img_path).convert('RGB')
            # print(type(img))
            img = get_test_transform(
                mean, std, size=cfg.INPUT_SIZE)(img).unsqueeze(0)

            if dev == 1:
                img = img.cuda()
            with torch.no_grad():
                out = model(img)
            prediction = torch.argmax(out, dim=1).cpu().item()
            pred_list.append(prediction)

    return _id, pred_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='', help='test data path')  # 测试数据的根目录位置
    parser.add_argument('--train_path', type=str, default='', help='train data path')  # 训练数据集的根目录位置
    parser.add_argument('--model', type=str, default='', help='trained model path')
    parser.add_argument('--model_name', type=str, default='resnet18',
                        help='The exact name of the model in torchvision')  # such as DenseNet, resnet152, vgg11, mobilenet_v3_small

    args = parser.parse_args()

    if not os.path.exists('./infdata/'):
        os.makedirs('./infdata/')
    if not os.path.exists('./result/'):
        os.makedirs('./result/')
    save_path = './result/{}_result.csv'.format(args.model_name)

    with open(cfg.TEST_LABEL_DIR, 'r') as f:
        imgs = f.readlines()

    # _id, pred_list = tta_predict(trained_model)
    idx, pred_list = predict(args.model, args.model_name, test_percent=1)
    Truth = get_labels(args.test_path)  # 文件名表示了真实标签
    Class_name = get_pre(args.train_path)  # 训练时给定的预测标签
    pre_res = []
    for i in pred_list:
        if int(i) == 1:
            pre_res.append(Class_name[1])
        else:
            pre_res.append(Class_name[0])

    submission = pd.DataFrame({'id': idx, 'labels': pre_res})
    submission.to_csv('./infdata/{}_submission.csv'.format(args.model_name), index=False)

    data = deal_csv('./infdata/{}_submission.csv'.format(args.model_name).format(args.model_name), Truth, Class_name)

    plot_cm(data)
    auc = plot_roc(data)

    acc, recall, precision, f1 = evaluate(data)

    result = make_csv(args.model_name, acc, recall, precision, f1, auc)
    result.to_csv(save_path, header=True, mode='a')
