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

import statistics





if __name__ == "__main__":
    # Creating a sample of data
    # sample = [0.867, 0.868, 0.87, 0.867, 0.867]  # DesnetCT
    # sample = [0.927, 0.924, 0.927, 0.926] # Resnet152xray
    # sample = [0.882 ,  0.887  ,   0.893  ,   0.884  , 0.893] # Resnet152xray
    # sample = [0.961 , 0.959 , 0.961 , 0.949 , 0.963] # jdcnetxray
    # sample = [0.951 ,  0.953  ,   0.957 ,   0.949  , 0.951] # jdcnetCT
    # sample = [0.941 , 0.943  ,   0.939   , 0.941 ] # Res101CT
    # sample = [0.927, 0.924, 0.927, 0.926] # Resnet152xray
    sample = [0.931 , 0.93 , 0.929 , 0.931 , 0.93] # SqueezenetXray
    sample = [0.867 , 0.868 , 0.87 , 0.867 , 0.867] # DesnetXRAY
    sample = [0.961 ,  0.961 , 0.963 ,  0.96 , 0.962 ] # GOOGLENETCT
    sample = [0.937 ,  0.934 , 0.936 ,  0.935 , 0.935] # RESNET152CT
    sample = [0.93 , 0.936 , 0.936 , 0.934 ,  0.937 ] # RESNET101XR
    sample = [0.953 , 0.956 , 0.953 , 0.954 , 0.953] # RESNET18XR
    sample = [0.97 , 0.97 , 0.97 , 0.968 , 0.973] # RESNET18CT
    sample = [0.979 ,0.981 ,0.977 , 0.98  , 0.98] # RegNetXR
    sample = [0.995 ,0.995 ,0.995 , 0.995  , 0.994] # CTINC
    sample = [0.989 ,0.989 ,0.989 , 0.989  , 0.99] # XRINC

    # Prints variance of the sample set

    # Function will automatically calculate
    # it's mean and set it as xbar
    print("Variance of sample set is % s"
          % (statistics.variance(sample)))

    print("Standard Deviation of sample is % s "
          % (statistics.stdev(sample)))