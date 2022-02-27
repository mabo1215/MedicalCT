import os
##class related
# NUM_CLASSES = 206
# BATCH_SIZE = 32

## default image size for model
INPUT_SIZE = 640
## epoch Num.
# MAX_EPOCH = 100

# WEIGHT_DECAY = 5e-4
# MOMENTUM = 0.9
## default learning rate
# LR = 1e-3


## Model Name

# model_name = 'DenseNet'
# model_name = 'Resnet152'
# model_name = 'Resnet18'
# model_name = 'Mobilenet_v3_small'
# model_name = 'SqueezeNet'
# model_name = 'Vgg'
# model_name = 'Googlenet'
# model_name = 'Shufflenetv2'
# model_name = 'Efficientnet'
# model_name = 'Mnasnet1_0'
# model_name = 'Regnet'
# model_name = 'Alexnet'
model_name = 'Inceptionv3'
# model_name = 'Resnet101'



# weights_path = 'weights/resnext101_32x32d/shuffle_10.pth'
# weights_path = 'checkpoint/CodeDes/model_0100.pth'
# weights_path = 'checkpoint/CTSqeExp/model_0100.pth'
# weights_path = 'checkpoint/CTVggExp/model_0100.pth'
# weights_path = 'checkpoint/XrayRegExp/XrayReg_0100.pth'
# weights_path = 'checkpoint/XrayRegExp/XrayReg_0100.pth'
# weights_path = 'checkpoint/CTIncExp/CTInc_0100.pth'
# weights_path = 'checkpoint/XrayIncExp/XrayInc_best.pth'
weights_path = 'checkpoint/AmazonIncExp/amainc_best.pth'
# weights_path = 'checkpoint/AmazonGogExp/AmazonGog_best.pth'


# from imgmodels import Resnet50, Resnet101, Resnext101_32x8d,Resnext101_32x16d, Densenet121, Densenet169, Mobilenetv2, Efficientnet, Resnext101_32x32d, Resnext101_32x48d
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


## Model saved path
SAVE_FOLDER = os.path.join(BASE, 'weights/')

## data path
# TRAIN_LABEL_DIR =BASE + 'train.txt'
# VAL_LABEL_DIR = BASE + 'val.txt'
# TEST_LABEL_DIR = os.path.join(BASE,  'infdata/test.txt')
# TEST_LABEL_DIR = 'E:/work/2/CT/COVID19Dataset/Xray/test.txt'
TEST_LABEL_DIR = "E:/work/2/imgdir/amazon.txt"

## full saved path for trained_model
TRAINED_MODEL = os.path.join(BASE, weights_path)




