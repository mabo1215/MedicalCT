import torch
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter
import cfg
import random
import math
import torch
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms, models
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


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


def load_checkpoint(filepath, model_name):
    print(filepath)
    checkpoint = torch.load(filepath)
    if model_name == "DenseNet":
        model = models.DenseNet()
    elif model_name == "MobileNetV3":
        model = models.MobileNetV3()
    elif model_name == "ShuffleNetV2":
        model = models.ShuffleNetV2()
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    else:
        model = models.resnet101(pretrained=True)
    # model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint)  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def result_csv(csv_url, truth, predict, model_name):
    data = pd.read_csv(csv_url)
    data.columns = ['id', 'labels']
    dtp = data[data.id.str.contains(str(truth[0])) & data.labels.str.contains(str(predict[0]))]
    dfn = data[data.id.str.contains(str(truth[0])) & data.labels.str.contains(str(predict[1]))]
    dfp = data[data.id.str.contains(str(truth[1])) & data.labels.str.contains(str(predict[0]))]
    dtn = data[data.id.str.contains(str(truth[1])) & data.labels.str.contains(str(predict[1]))]
    tp = len(dtp)
    fp = len(dfp)
    tn = len(dtn)
    fn = len(dfn)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    # model_name = [str(model_name)]
    acc = round((tp + tn) / (tp + fp + fn + tn), 3)
    recall = round(tp / (tp + fn), 3)
    precision = round(tp / (tp + fp), 3)
    false_positive_rate = round(fp / (fp + tn + 0.01), 3)
    positive_predictive_value = round(tp / (tp + fp + 0.01), 3)
    negative_predictive_value = round(tn / (fn + tn + 0.01), 3)
    result = pd.DataFrame({"model name": model_name,
                           "Accuracy": acc,
                           "Recall": recall,
                           "Precision": precision,
                           # "Variance": statistics.variance(),
                           "False positive rate": false_positive_rate,
                           "Positive predictive value": positive_predictive_value,
                           "Negative predictive value": negative_predictive_value})

    # return result
    return result, tpr, fpr


def predict(model, model_name):
    # 读入模型
    model = load_checkpoint(model, model_name)
    print('..... Finished loading model! ......')
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ##将模型放置在gpu上运行
    dev = 0
    if torch.cuda.is_available():
        dev = 1
    if dev == 1:
        model.cuda()
    pred_list, _id = [], []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        # print(img_path)
        _id.append(os.path.basename(img_path).split('.')[0])
        img = Image.open(img_path).convert('RGB')
        # print(type(img))
        img = get_test_transform(mean, std, size=cfg.INPUT_SIZE)(img).unsqueeze(0)

        if dev == 1:
            img = img.cuda()
        with torch.no_grad():
            out = model(img)
        prediction = torch.argmax(out, dim=1).cpu().item()
        pred_list.append(prediction)
    return _id, pred_list


if __name__ == "__main__":
    trained_model = cfg.TRAINED_MODEL
    model_name = cfg.model_name
    save_path = './infdata/{}_result.csv'.format(model_name)
    with open(cfg.TEST_LABEL_DIR, 'r') as f:
        imgs = f.readlines()

    # _id, pred_list = tta_predict(trained_model)
    idx, pred_list = predict(trained_model, model_name)
    # print(pred_list)
    # print(idx)
    Truth = ['non', 'covid']
    Class_name = ['Normal', 'Pneumonia']
    pre_res = []
    for i in pred_list:
        if int(i) == 1:
            pre_res.append(Class_name[0])
        else:
            pre_res.append(Class_name[1])
    # print(pre_res)
    idx = list(idx)
    submission = pd.DataFrame({'id': idx, 'labels': pre_res})
    if not os.path.exists(cfg.BASE + '/infdata/'):
        os.makedirs(cfg.BASE + '/infdata/')
    submission.to_csv(cfg.BASE + '/infdata/{}_submission.csv'
                      .format(model_name), index=False)

    # result = result_csv(cfg.BASE + '/infdata/{}_submission.csv'.format(model_name), Truth, Class_name, model_name)
    # result.to_csv(save_path,index=False, mode='a')
    result, tpr, fpr = result_csv(cfg.BASE + '/infdata/{}_submission.csv'.format(model_name), Truth, Class_name,
                                  model_name)
    # plot_roc_curve(fpr, tpr)
