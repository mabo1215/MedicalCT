import transformers
from transformers import AutoFeatureExtractor, DeiTForImageClassification ,DeiTFeatureExtractor , ViTForImageClassification , ViTFeatureExtractor , ViTModel
from PIL import Image
import requests
import torch
import argparse
from torchvision import datasets
from torchvision import transforms as tfc
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import tqdm
import os
import numpy as np
import math
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from utils.utils import *
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from hugsvision.dataio.VisionDataset import VisionDataset
import matplotlib
matplotlib.use('TkAgg')

# 对训练集做一个变换
# train_transforms = tfc.Compose([
#     tfc.RandomResizedCrop(224),  # 对图片尺寸做一个缩放切割
#     tfc.RandomHorizontalFlip(),  # 水平翻转
#     tfc.ToTensor(),  # 转化为张量
#     tfc.Normalize((.5, .5, .5), (.5, .5, .5))  # 进行归一化
# ])

# 对测试集做变换
test_transforms = tfc.Compose([
    tfc.RandomResizedCrop(224),
    tfc.ToTensor(),
    tfc.Normalize((.5, .5, .5), (.5, .5, .5))
])

# def get_train_transform():
#     return tfc.Compose([
#         tfc.RandomResizedCrop(224),
#         tfc.RandomHorizontalFlip(),  # 水平翻转
#         tfc.ToTensor(),
#         tfc.Normalize((.5, .5, .5), (.5, .5, .5))
#     ])

def get_test_transform(mean, std, size=0):
    return tfc.Compose([
        tfc.RandomResizedCrop(224),
        tfc.ToTensor(),
        tfc.Normalize((.5, .5, .5), (.5, .5, .5))
    ])


def prediction(sig_img,model_name,model_dir):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    im = Image.open(sig_img).convert('RGB')

    plt.imshow(im)
    im.save(sig_img)

    # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # img = get_test_transform(mean, std, size=640)(im).unsqueeze(0)
    # # model_state_dict = torch.load(model_dir)
    # checkpoint = torch.load(model_dir)


    m_d = f'{model_dir}/model/'
    json_d = f'{model_dir}/feature_extractor/'
    # print(m_d)

    if model_name == 'DeiT':
        classifier = VisionClassifierInference(
            feature_extractor=DeiTFeatureExtractor.from_pretrained(model_dir),
            model=DeiTForImageClassification.from_pretrained(model_dir),
        )
    else:
        classifier = VisionClassifierInference(
            feature_extractor=ViTFeatureExtractor.from_pretrained(json_d),
            model=ViTForImageClassification.from_pretrained(m_d),
        )

    print(sig_img)
    label = classifier.predict(img_path=sig_img)
    print("Predicted class:", label)

    plt.show()

def evaluate(img_pathlist,model_name,model_dir,resout_dir):
    acc_num = 0
    img_num = 0
    m_d = f'{model_dir}/model/'
    json_d = f'{model_dir}/feature_extractor/'

    if model_name == 'DeiT':
        classifier = VisionClassifierInference(
            feature_extractor=DeiTFeatureExtractor.from_pretrained(json_d),
            model=DeiTForImageClassification.from_pretrained(m_d),
        )
    else:
        classifier = VisionClassifierInference(
            feature_extractor=ViTFeatureExtractor.from_pretrained(json_d),
            model=ViTForImageClassification.from_pretrained(m_d),
        )

    save_csv_res = resout_dir + '/infdata/{}_submission.csv'.format(model_name)
    # print(save_csv_res)
    with open(img_pathlist,'r+') as txtfile:
        with open(save_csv_res, 'w+', encoding='utf-8') as csvfile:
            spam = txtfile.readlines()
            for ful_path in tqdm.tqdm(spam):
                ful_path = ful_path.replace('\n','')
                im = Image.open(ful_path).convert('RGB')
                # plt.imshow(im)
                im.save(ful_path)

                label = classifier.predict(img_path=ful_path)

                file_base_name = os.path.basename(ful_path)

                file_base_name_fr = str(file_base_name.split('.')[0])

                csvfile.writelines([file_base_name_fr+','+label+'\n'])
                # print(file_base_name,file_base_name.find('Cov'))

                if label == 'COVID' and file_base_name.find('Cov')>-1:
                    acc_num+=1
                elif label == 'NORMAL' and file_base_name.find('Norm')>-1:
                    acc_num+=1
                img_num+=1
            acc_rate = acc_num/img_num
            print(f'Predicted acc rate :{acc_rate}')
        csvfile.close()
    txtfile.close()
        # print(f'Predicted class:{label},  the file name :{file_base_name}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="E:/work/2/CT/chest_xray/",help="")  #E:\work\2\CT\chest_xray   # E:/work/2/CT/COVID19Dataset/Xray/
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default="E:/source/MedicalCT/CTBob/checkpoint/XRAYDEITEXP/100_2022-02-16-22-21-26/",help="")   # XrSquExp, CTSqeExp , CTVggExp, CodeDesExp ,CTRen152Exp , XraySqeExp , CTGOOGLExp
    parser.add_argument("--no-cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--model-name', type=str, default="DeiT",help="")  # DenseNet, resnet101 , resnet152 ,Vgg, SqueezeNet , CTvggExp ,Transformer ,googlenet, resnet18  , Vit , DeiT
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--infimg', type=str, default="E:/work/2/CT/res/",help="") #Norm5478.jpg , Cov937.jpg , Cov2945.jpg
    parser.add_argument('--batchinflist', type=str, default="E:/work/2/CT/chest_xray/test.txt",help="") #Norm5478.jpg , Cov937.jpg , Cov2945.jpg
    parser.add_argument('--resoutdir', type=str, default="E:/source/MedicalCT/CTBob",help="") #Norm5478.jpg , Cov937.jpg , Cov2945.jpg


    args = parser.parse_args()

    # 定义损失函数和优化器
    dataset_dir = args.data_dir
    ratio = args.ratio
    batch_size = args.batch_size
    lr, num_epochs = args.lr, args.epochs
    model_name = args.model_name

    dev = 1 if torch.cuda.is_available() and not args.no_cuda else 0

    sig_img = args.infimg
    batchinflist = args.batchinflist
    model_dir = args.model_dir
    resoutdir = args.resoutdir
    # prediction(sig_img,model_name,model_dir)
    evaluate(batchinflist,model_name,model_dir,args.resoutdir)

    print(resoutdir + '/infdata/{}_submission.csv')

    Truth = ['COVID','NORMAL']
    Class_name = ['COVID','NORMAL']
    data = deal_csv(resoutdir + '/infdata/{}_submission.csv'.format(model_name), Truth, Class_name)
    auc = plot_roc(data)
    plot_cm(data)
    acc, recall, precision, f1 = evaluate_res(data)

    result = make_csv(model_name, acc, recall, precision, f1, auc)
    result.to_csv(resoutdir+'/infdata/res{}.csv'.format(model_name), header=True, mode='a+')
