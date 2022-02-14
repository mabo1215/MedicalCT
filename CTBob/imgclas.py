import os.path
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
# from model import ctmodel
from transformers import ViTFeatureExtractor, ViTModel


#对训练集做一个变换
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),		#对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),		#水平翻转
    transforms.ToTensor(),					#转化为张量
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))	#进行归一化
])

#对测试集做变换
test_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

def load_model(model_name,dev,lr):
    net = []
    loss = 0
    optimizer = []
    if model_name == 'resnet101':
        net = models.resnet101(pretrained=True)
        # net = ResNet.ResNet_model(50).cuda()  #使用GPU训练
        # net = torch.nn.DataParallel(ResNet.net).cuda()  #使用多块GPU共同训练
    elif model_name == "ShuffleNetV2":
        net = models.ShuffleNetV2(pretrained=True,width_mult=1.0,last_channel=True)
    elif model_name == "DenseNet":
        net = models.DenseNet()
    elif model_name == "resnet152":
        net = models.resnet152(pretrained=True)
    elif model_name == "resnet18":
        net = models.resnet18(pretrained=True)
    elif model_name == "MobileNetV3":
        net = models.MobileNetV3()
    elif model_name == "SqueezeNet":
        net = models.squeezenet1_1(pretrained=True)
    elif model_name == "Vgg":
        net = models.vgg16(pretrained=True)
    elif model_name == "googlenet":
        net = models.googlenet(pretrained=True,progress =  True)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)  # 优化器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # lr scheduler

    if dev == 1 and model_name != "ctmodel":
        net.to('cuda')
        loss = torch.nn.CrossEntropyLoss().to('cuda')  # 损失函数
    elif dev == 0 and model_name != "ctmodel":
        # torch.device('cpu')
        loss = torch.nn.CrossEntropyLoss()  # 损失函数
    print("model loaded...")
    return net,loss,optimizer,scheduler

def load_local_dataset(dataset_dir, ratio = 0.8, batch_size = 256):
    #获取数据集
    all_datasets = datasets.ImageFolder(dataset_dir, transform=train_transforms)
    #将数据集划分成训练集和测试集
    train_size=int(ratio * len(all_datasets))
    test_size=len(all_datasets) - train_size
    train_datasets, test_datasets = torch.utils.data.random_split(all_datasets, [train_size, test_size])

    train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

    return train_iter,test_iter

def load_train_test_dataset(train_dir, test_dir , batch_size = 256):
    #获取数据集
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

    return train_iter,test_iter

def load_datasets(dataset_dir):
    # 训练集和测试集在一个文件夹下

    train_iter, test_iter = load_local_dataset(dataset_dir, ratio, batch_size)

    print("data loaded...")
    print("train-sets=", len(train_iter))
    print("val-sets=", len(test_iter))
    return train_iter, test_iter


#训练模型
def train(net, train_iter, test_iter, optimizer,  loss, num_epochs,dev,save_dir, scheduler):
    for epoch in range(num_epochs):
        # 训练过程
        net.train()  # 启用 BatchNormalization 和 Dropout
        train_l_sum, train_acc_sum, train_num = 0.0, 0.0, 0
        for X, y in train_iter:
            if dev == 1:
                X = X.to('cuda')
                y = y.to('cuda')
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            #计算准确率
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            train_num += y.shape[0]
        scheduler.step()
        print('epoch %d, loss %.4f, train acc %.3f' % (epoch + 1, train_l_sum / train_num, train_acc_sum / train_num))

        # 测试过程
        if (epoch+1) %10 == 0:
            test_acc_sum, test_num= 0.0, 0
            with torch.no_grad(): #不求梯度、反向传播
                net.eval()  # 不启用 BatchNormalization 和 Dropout
                for X,y in test_iter:
                    if dev == 1:
                        X = X.to('cuda')
                        y = y.to('cuda')
                    test_acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                    test_num += y.shape[0]
                print('test acc %.3f' % (test_acc_sum / test_num))

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(net.state_dict(), f'{save_dir}/model_{str(epoch + 1).zfill(4)}.pth')  # 保存模型

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="E:/work/2/CT/COVID19Dataset/XRay/",help="")
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    # Data, model, and output directories
    parser.add_argument('--save-dir', type=str, default="E:/source/MedicalCT/CTBob/checkpoint/XraySqeExp/",help="")   # XrSquExp, CTSqeExp , CTVggExp, CodeDesExp ,CTRen152Exp , XraySqeExp , CTGOOGLExp
    parser.add_argument("--no-cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--model-name', type=str, default="SqueezeNet",help="")  # DenseNet, resnet101 , resnet152 ,Vgg, SqueezeNet , CTvggExp ,Transformer ,googlenet, resnet18

    args = parser.parse_args()

    # 定义损失函数和优化器
    dataset_dir = args.data_dir
    ratio = args.ratio
    batch_size = args.batch_size
    lr, num_epochs = args.lr, args.epochs

    dev = 1 if torch.cuda.is_available() and not args.no_cuda else 0

    print(args,dev)
    train_iter, test_iter = load_local_dataset(dataset_dir,ratio,batch_size)
    net, loss, optimizer, scheduler = load_model(args.model_name,dev,lr)
    train(net, train_iter, test_iter, optimizer, loss, num_epochs,dev,args.save_dir, scheduler)
