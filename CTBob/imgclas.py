import os.path
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import tqdm
from torchvision import datasets, transforms
# from model import ctmodel
from transformers import ViTFeatureExtractor, ViTModel
import wandb
import logging



logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

#对训练集做一个变换
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(299),		#对图片尺寸做一个缩放切割 224
    transforms.RandomHorizontalFlip(),		#水平翻转
    transforms.ToTensor(),					#转化为张量
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))	#进行归一化
])

#对训练集做一个变换 inceptionv3
train_transforms_inc = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#对训练集做一个变换
test_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),		#对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),		#水平翻转
    transforms.ToTensor(),					#转化为张量
    transforms.Normalize((.3, .3, .3), (.3, .3, .3))	#进行归一化
])

#对训练集做一个变换 inceptionv3
test_transforms_inc = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(model_name,dev,lr,pretrain,progress):
    net = []
    loss = 0
    optimizer = []
    if model_name == 'Resnet101':
        net = models.resnet101(pretrained=pretrain,)
        # net = ResNet.ResNet_model(50).cuda()  #使用GPU训练
        # net = torch.nn.DataParallel(ResNet.net).cuda()  #使用多块GPU共同训练
    elif model_name == "DenseNet":
        net = models.DenseNet()
    elif model_name == "Resnet152":
        net = models.resnet152(pretrained=pretrain)
    elif model_name == "Resnet18":
        net = models.resnet18(pretrained=pretrain)
    elif model_name == "Mobilenet_v3_small":
        net = models.mobilenet_v3_small(pretrained=pretrain,progress=progress)
    elif model_name == "SqueezeNet":
        net = models.squeezenet1_1(pretrained=pretrain)
    elif model_name == "Vgg":
        net = models.vgg11(pretrained=pretrain)
    elif model_name == "Googlenet":
        net = models.googlenet(pretrained=pretrain,progress=progress)
    elif model_name == "Shufflenetv2":
        net = models.shufflenet_v2_x1_0(pretrained=pretrain)
    elif model_name == "Efficientnet":
        net = models.efficientnet_b7(pretrained=pretrain)
    elif model_name == "Mnasnet1_0":
        net = models.mnasnet1_0(pretrained=pretrain,progress=progress)
    elif model_name == "Regnet":
        net = models.regnet_y_800mf(pretrained=pretrain,progress=progress)
    elif model_name == "Alexnet":
        net = models.alexnet(pretrained=pretrain)
    elif model_name == "Inceptionv3":
        net = models.inception_v3(pretrained=pretrain)
        print("Load Inceptionv3 model")
    else:
        net = models.__dict__[model_name](pretrained=pretrain, progress=progress)
    # if model_name == "Alexnet":
    #     optimizer = torch.optim.sgd(net.parameters(), lr=lr)  # 优化器
    # else:
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 优化器
    # optimizer = torch.optim.AdamW(net.parameters(), lr=lr)  # 优化器
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)  # 优化器
    optimizer = torch.optim.ASGD(net.parameters(), lr=lr)  # 优化器
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # lr scheduler
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr= 0.3 , last_epoch=-1)  # lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)  # lr scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult=1, eta_min=0.1, last_epoch=-1)  # lr scheduler

    if dev == 1 and model_name != "ctmodel":
        net.to('cuda')
        loss = torch.nn.CrossEntropyLoss().to('cuda')  # 损失函数
    elif dev == 0 and model_name != "ctmodel":
        # torch.device('cpu')
        loss = torch.nn.CrossEntropyLoss()  # 损失函数
    # elif model_name == "Inceptionv3":
    #     # torch.device('cpu')
    #     loss = torch.nn.Softmax().to('cuda')  # 损失函数
    print("model loaded...")
    return net,loss,optimizer,scheduler

def load_local_dataset(dataset_dir, model_name, ratio = 0.8, batch_size = 256):
    #获取数据集
    if model_name == 'Inceptionv3':
        all_datasets = datasets.ImageFolder(dataset_dir, transform=train_transforms_inc)
        print("Load dataset with inception format")
    else:
        all_datasets = datasets.ImageFolder(dataset_dir, transform=train_transforms)
    class_list = all_datasets.classes

    with open(dataset_dir+'class_list.txt','w') as f:
        for i in class_list:
            f.writelines(i+'\n')
    f.close()
    #将数据集划分成训练集和测试集
    train_size=int(ratio * len(all_datasets))
    test_size=len(all_datasets) - train_size
    train_datasets, test_datasets = torch.utils.data.random_split(all_datasets, [train_size, test_size])

    train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

    return train_iter,test_iter,class_list

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
def train(net, train_iter, test_iter, optimizer,  loss, num_epochs,dev,save_dir, scheduler,model_name):
    best_acc = 0.0
    for epoch in range(num_epochs):
        # 训练过程
        net.train()  # 启用 BatchNormalization 和 Dropout
        # wandb.watch(net)
        train_l_sum, train_acc_sum, train_num = 0.0, 0.0, 0
        for X, y in tqdm.tqdm(train_iter):
            if dev == 1:
                X = X.to('cuda')
                y = y.to('cuda')
            if model_name == 'Inceptionv3':
                y_hat = net(X).logits
            else:
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
        loss_rate = train_l_sum / train_num
        acc_rate = train_acc_sum / train_num
        print('\n epoch %d, loss %.4f, train acc %.3f ' % (epoch + 1, loss_rate, acc_rate))
        wandb.log({'epoch': epoch, 'loss': float(loss_rate*8000), 'accuracy': acc_rate,'train number': train_num, 'train_l_sum': train_l_sum})

        # 测试过程
        if (epoch+1) %10 == 0:
            # wandb.save(save_dir + f'save_{epoch}.h5')
            wandb.watch(net)
            test_acc_sum, test_num= 0.0, 0
            tmp_acc = 0.0
            with torch.no_grad(): #不求梯度、反向传播
                net.eval()  # 不启用 BatchNormalization 和 Dropout
                for X,y in test_iter:
                    if dev == 1:
                        X = X.to('cuda')
                        y = y.to('cuda')
                    test_acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                    test_num += y.shape[0]
                tmp_acc = test_acc_sum / test_num
                print('test acc %.3f' % (tmp_acc))

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(net.state_dict(), f'{save_dir}/model_{str(epoch + 1).zfill(4)}.pth')  # 保存模型

            if tmp_acc> best_acc:
                best_acc = tmp_acc
                torch.save(net.state_dict(), f'{save_dir}/model_best.pth')  # 保存模型

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="E:/work/2/CT/COVID19Dataset/Xray/",help="") #E:/work/2/CT/COVID19Dataset/Xray/   E:/work/2/CT/COVID19Dataset/CT/ E:/work/2/imgdir/amazon.txt  E:/work/2/imgdir/
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    # Data, model, and output directories
    parser.add_argument('--save-dir', type=str, default="E:/source/MedicalCT/CTBob/checkpoint/XrMnaExp/",help="")   #AmazonDenExp, XrSquExp, CTSqeExp , CTVggExp, CodeDesExp ,CTRen152Exp , XraySqeExp , CTGOOGLExp, XrGOOGLExp, XrayIncExp AmazonGogExp AmazonIncExp
    parser.add_argument("--no-cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--pretrain", action="store_true", help="Pretrain network")
    parser.add_argument("--progress", action="store_true", help="progress network")
    parser.add_argument('--model-name', type=str, default="regnet_x_32gf",
                        help="")  # mnasnet0_75, DenseNet, Resnet101 , Resnet152 ,Vgg, SqueezeNet , CTvggExp ,Transformer ,Googlenet, Resnet18 , Mobilenet_v3_small ,Shufflenetv2 , Vgg , Inceptionv3 ,Regnet , Alexnet ,Efficientnet , Mnasnet1_0

    args = parser.parse_args()

    # 定义损失函数和优化器
    dataset_dir = args.data_dir
    ratio = args.ratio
    batch_size = args.batch_size
    lr, num_epochs = args.lr, args.epochs
    model_name = args.model_name
    save_dir = args.save_dir
    pretrain = args.pretrain
    progress = args.progress

    dev = 1 if torch.cuda.is_available() and not args.no_cuda else 0

    print(args,dev)
    train_iter, test_iter, class_list = load_local_dataset(dataset_dir, model_name,ratio,batch_size)
    net, loss, optimizer, scheduler = load_model(model_name,dev,lr,pretrain,progress)

    wandb.init(project=model_name, entity="mabo1215")

    wandb.config = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size
    }

    train(net, train_iter, test_iter, optimizer, loss, num_epochs,dev,save_dir, scheduler,model_name)
