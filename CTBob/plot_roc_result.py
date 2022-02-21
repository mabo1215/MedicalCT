import os
import numpy as np
import pandas as pd
from utils.utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import warnings

warnings.filterwarnings("ignore")

data_path = data_path = ''
file = os.listdir(data_path)
data0 = deal_csv(data_path + file[0], 'id', 'labels')
data1 = deal_csv(data_path + file[1], 'id', 'labels')
data2 = deal_csv(data_path + file[2], 'id', 'labels')
data3 = deal_csv(data_path + file[3], 'id', 'labels')
data4 = deal_csv(data_path + file[4], 'id', 'labels')
data5 = deal_csv(data_path + file[5], 'id', 'labels')
# data1 = pd.read_csv(data_path + file[1])
tandf0, auc0 = eval_roc(data0)
tandf1, auc1 = eval_roc(data1)
tandf2, auc2 = eval_roc(data2)
tandf3, auc3 = eval_roc(data3)
tandf4, auc4 = eval_roc(data4)
tandf5, auc5 = eval_roc(data5)
name = []
for i in file:
    name.append(i[:-15])
plt.figure(figsize=(10, 10), dpi=100)

# colorname = ['bule', 'black']


plt.plot(tandf0['TP'], tandf0['FP'], label='{} (AUC={:.3f})'.format(name[0], auc0), color='g')
plt.plot(tandf1['TP'], tandf1['FP'], label='{} (AUC={:.3f})'.format(name[1], auc0), color='b')
plt.plot(tandf2['TP'], tandf2['FP'], label='{} (AUC={:.3f})'.format(name[2], auc0), color='r')
plt.plot(tandf3['TP'], tandf3['FP'], label='{} (AUC={:.3f})'.format(name[3], auc0), color='k')
plt.plot(tandf4['TP'], tandf4['FP'], label='{} (AUC={:.3f})'.format(name[4], auc0), color='y')
plt.plot(tandf5['TP'], tandf5['FP'], label='{} (AUC={:.3f})'.format(name[5], auc0), color='pink')
plt.legend(loc='lower right')
plt.title('Receiver Operating Characteristic')
plt.plot([(0, 0), (1, 1)], 'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
