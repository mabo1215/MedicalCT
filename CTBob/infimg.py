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
from torchvision import transforms,models
from transformers import ViTFeatureExtractor, ViTModel



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
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

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

def get_test_transform_inception(mean, std, size=0):
    return transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_checkpoint(filepath,model_name):
    print(filepath)
    checkpoint = torch.load(filepath)
    if model_name == "DenseNet":
        model = models.DenseNet()
    elif model_name == "Resnet152":
        net = models.resnet152(pretrained=True)
    elif model_name == "Resnet18":
        net = models.resnet18(pretrained=True)
    elif model_name == "Mobilenet_v3_small":
        net = models.mobilenet_v3_small(pretrained=True, progress=True)
    elif model_name == "SqueezeNet":
        net = models.squeezenet1_1(pretrained=True)
    elif model_name == "Vgg":
        net = models.vgg11(pretrained=True)
    elif model_name == "Googlenet":
        net = models.googlenet(pretrained=True, progress=True)
    elif model_name == "Shufflenetv2":
        net = models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == "Efficientnet":
        net = models.efficientnet_b7(pretrained=True)
    elif model_name == "Mnasnet1_0":
        net = models.mnasnet1_0(pretrained=True, progress=True)
    elif model_name == "Regnet":
        net = models.regnet_y_800mf(pretrained=True, progress=True)
    elif model_name == "Alexnet":
        net = models.alexnet(pretrained=True)
    elif model_name == "Inceptionv3":
        net = models.inception_v3(pretrained=True)
    else:
        model = models.resnet101(pretrained=True)
    # model = checkpoint['model']  # extra model
    model.load_state_dict(checkpoint)  # load model
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def predict(model,model_name):
    # read models
    model = load_checkpoint(model,model_name)
    print('..... Finished loading model! ......')
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ##in gpu
    dev =0
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
        if model_name =="Inceptionv3":
            img = get_test_transform_inception(mean,std,size=cfg.INPUT_SIZE)(img).unsqueeze(0)
        else:
            img = get_test_transform(mean,std,size=cfg.INPUT_SIZE)(img).unsqueeze(0)

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
    with open(cfg.TEST_LABEL_DIR,  'r')as f:
        imgs = f.readlines()

    # _id, pred_list = tta_predict(trained_model)
    _id, pred_list = predict(trained_model,model_name)

    Class_name= ['Covid','Normal']
    pre_res = []
    for i in tqdm.tqdm(pred_list):
        if int(i) == 1:
            pre_res.append(Class_name[-1])
        else:
            pre_res.append(Class_name[0])

    print(_id,pre_res)

    submission = pd.DataFrame({"ID": _id, "Label": pre_res})
    submission.to_csv(cfg.BASE + '/infdata/{}_submission.csv'
                      .format(model_name), index=False, header=False)