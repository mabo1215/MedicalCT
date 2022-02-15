import transformers
from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher ,DeiTFeatureExtractor , ViTForImageClassification , ViTFeatureExtractor
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

class ImageClassificationCollator:
   def __init__(self, feature_extractor):
      self.feature_extractor = feature_extractor
   def __call__(self, batch):
      encodings = self.feature_extractor([x[0] for x in batch],
      return_tensors='pt')
      encodings['labels'] = torch.tensor([x[1] for x in batch],
      dtype=torch.long)
      return encodings


class Classifier(pl.LightningModule):
   def __init__(self, model, lr: float = 2e-5, **kwargs):
       super().__init__()
       self.save_hyperparameters('lr', *list(kwargs))
       self.model = model
       self.forward = self.model.forward
       self.val_acc = Accuracy()
   def training_step(self, batch, batch_idx):
       outputs = self(**batch)
       self.log(f"train_loss", outputs.loss)
       return outputs.loss
   def validation_step(self, batch, batch_idx):
       outputs = self(**batch)
       self.log(f"val_loss", outputs.loss)
       acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
       self.log(f"val_acc", acc, prog_bar=True)
       return outputs.loss
   def configure_optimizers(self):
       return torch.optim.Adam(self.parameters(),
                        lr=self.hparams.lr,weight_decay = 0.00025)

def load_local_dataset(dataset_dir, ratio = 0.8, batch_size = 256, model_name = 'Vit', num_workers = 1):
    #获取数据集
    # 对训练集做一个变换
    train_transforms = tfc.Compose([
        tfc.RandomResizedCrop(224),  # 对图片尺寸做一个缩放切割
        tfc.RandomHorizontalFlip(),  # 水平翻转
        tfc.ToTensor(),  # 转化为张量
        tfc.Normalize((.5, .5, .5), (.5, .5, .5))  # 进行归一化
    ])

    # 对测试集做变换
    test_transforms = tfc.Compose([
        tfc.RandomResizedCrop(224),
        tfc.ToTensor(),
        tfc.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    # all_datasets = datasets.ImageFolder(dataset_dir, transform=train_transforms)
    all_datasets = datasets.ImageFolder(dataset_dir, transform=train_transforms)

    indices = torch.randperm(len(all_datasets)).tolist()
    n_val = math.floor(len(indices) * (1- ratio))
    train_ds = torch.utils.data.Subset(all_datasets, indices[:-n_val])
    val_ds = torch.utils.data.Subset(all_datasets, indices[-n_val:])



    if model_name == 'DeiT':
        feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
    else:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    collator = ImageClassificationCollator(feature_extractor)

    train_iter = DataLoader(train_ds, batch_size=batch_size, collate_fn=collator, shuffle=True , num_workers=num_workers)
    val_ds = DataLoader(val_ds, batch_size=batch_size, collate_fn=collator, num_workers=num_workers)
    # test_iter = DataLoader(val_ds, batch_size=batch_size, collate_fn=collator, num_workers=num_workers)

    # print(train_iter)

    return train_iter,val_ds,all_datasets, feature_extractor



def load_model(model_name,dev,lr,all_datasets):
    loss = 0
    optimizer = []

    label2id = {}
    id2label = {}
    for i, class_name in enumerate(all_datasets.classes):
        label2id[class_name] = str(i)
        id2label[str(i)] = class_name



    if model_name == 'DeiT':
        model = DeiTForImageClassificationWithTeacher.from_pretrained(
            'facebook/deit-base-distilled-patch16-224',
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label)
    # elif model_name == "Vit":
    else:
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # 优化器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # lr scheduler

    if dev == 1:
        model.to('cuda')
        loss = torch.nn.CrossEntropyLoss().to('cuda')  # 损失函数
    elif dev == 0:
        # torch.device('cpu')
        loss = torch.nn.CrossEntropyLoss()  # 损失函数
    print("model loaded...")
    return model,loss,optimizer,scheduler

def train(model, train_iter, test_iter, optimizer,  loss, num_epochs,dev,save_dir, scheduler, all_datasets):
    pl.seed_everything(42)
    classifier = Classifier(model, lr=2e-5)
    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=num_epochs)
    trainer.fit(classifier, train_iter, test_iter)
    trainer.save_checkpoint(f'{save_dir}/model_{str(num_epochs).zfill(4)}.pth')

    # for epoch in range(num_epochs):
    #     # 训练过程
    #     model.train()  # 启用 BatchNormalization 和 Dropout
    #     train_l_sum, train_acc_sum, train_num = 0.0, 0.0, 0
    #     logits_and_label_list = []
    #
    #
    #     # for X, y in tqdm.tqdm(all_datasets.classes):
    #     #     model.train()
    #     #     if dev == 1:
    #     #         X = X.to('cuda')
    #     #         y = y.to('cuda')
    #     #     #     train_iter = train_iter.to('cuda')
    #     #     input_img = feature_extractor(images=train_iter, return_tensors="pt")
    #     #     y_hat = model(X)
    #     #
    #     #     outputs = model(**input_img)
    #     #     logits_and_label_list.append((outputs.logits, train_iter))
    #     #     l = outputs[0]
    #     #     optimizer.zero_grad()
    #     #     l.backward()
    #     #     optimizer.step()
    #     #     #计算准确率
    #     #     train_l_sum += l.item()
    #     #     train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
    #     #     train_num += y.shape[0]
    #     # scheduler.step()
    #     # print('\n epoch %d, loss %.4f, train acc %.3f ' % (epoch + 1, train_l_sum / train_num, train_acc_sum / train_num))
    #
    #     # 测试过程
    #     if (epoch+1) %10 == 0:
    #         test_acc_sum, test_num= 0.0, 0
    #         with torch.no_grad(): #不求梯度、反向传播
    #             model.eval()  # 不启用 BatchNormalization 和 Dropout
    #             for X,y in test_iter:
    #                 if dev == 1:
    #                     X = X.to('cuda')
    #                     y = y.to('cuda')
    #                 test_acc_sum += (model(X).argmax(dim=1) == y).float().sum().item()
    #                 test_num += y.shape[0]
    #             print('test acc %.3f' % (test_acc_sum / test_num))
    #
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         torch.save(model.state_dict(), f'{save_dir}/model_{str(epoch + 1).zfill(4)}.pth')  # 保存模型


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="E:/work/2/CT/COVID19Dataset/CT/",help="")
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    # Data, model, and output directories
    parser.add_argument('--save-dir', type=str, default="E:/source/MedicalCT/CTBob/checkpoint/CTRen18Exp/",help="")   # XrSquExp, CTSqeExp , CTVggExp, CodeDesExp ,CTRen152Exp , XraySqeExp , CTGOOGLExp
    parser.add_argument("--no-cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--model-name', type=str, default="Vit",help="")  # DenseNet, resnet101 , resnet152 ,Vgg, SqueezeNet , CTvggExp ,Transformer ,googlenet, resnet18
    parser.add_argument('--num-workers', type=int, default=2)


    args = parser.parse_args()

    # 定义损失函数和优化器
    dataset_dir = args.data_dir
    ratio = args.ratio
    batch_size = args.batch_size
    lr, num_epochs = args.lr, args.epochs

    dev = 1 if torch.cuda.is_available() and not args.no_cuda else 0

    print(args,dev)
    train_iter, test_iter,all_datasets, feature_extractor = load_local_dataset(dataset_dir,ratio,batch_size, args.model_name,args.num_workers)
    model, loss, optimizer, scheduler = load_model(args.model_name,dev,lr,all_datasets)
    train(model, train_iter, test_iter, optimizer, loss, num_epochs,dev,args.save_dir, scheduler, all_datasets)


    url = 'E:/source/MedicalCT/CTBob/infdata/n333.jpeg'
    im = Image.open(url)
    inputs = feature_extractor(images=im, return_tensors="pt")
    inputs.keys()

    pixel_values = inputs['pixel_values']

    # forward pass
    outputs = model(pixel_values)
    logits = outputs.logits

    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])