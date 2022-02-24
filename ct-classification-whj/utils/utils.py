import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import warnings

warnings.filterwarnings("ignore")


def deal_csv(csv_url, ids, labels):
    data = pd.read_csv(csv_url)
    data.columns = ['truth', 'predict']
    data['truth'] = data['truth'].apply(lambda x: 1 if ids[0] in x else 0)
    data['predict'] = data['predict'].apply(lambda x: 1 if labels[0] in x else 0)
    data.sort_values('predict', inplace=True, ascending=False)

    return data


def plot_cm(data):
    cm = np.arange(4).reshape(2, 2)
    cm[0, 0] = len(data[(data.truth == 0) & (data.predict == 0)])
    cm[0, 1] = len(data[(data.truth == 0) & (data.predict == 1)])
    cm[1, 0] = len(data[(data.truth == 1) & (data.predict == 0)])
    cm[1, 1] = len(data[(data.truth == 1) & (data.predict == 1)])
    import itertools
    classes = [0, 1]
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="red" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    plt.show()


def plot_roc(data):
    data.sort_values('predict', inplace=True, ascending=False)
    TPRandFPR = pd.DataFrame(index=range(len(data)), columns=('TP', 'FP'))
    for j in range(len(data)):
        data1 = data.head(n=j + 1)
        FP = len(data1[data1['truth'] == 0]) / float(len(data[data['truth'] == 0]))
        TP = len(data1[data1['truth'] == 1]) / float(len(data[data['truth'] == 1]))
        TPRandFPR.iloc[j] = [TP, FP]
    AUC = auc(TPRandFPR['FP'], TPRandFPR['TP'])
    # plt.scatter(x=TPRandFPR['FP'],y=TPRandFPR['TP'],label='(FPR,TPR)',color='b')
    plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], 'k', label='AUC = %0.2f' % AUC)
    plt.legend(loc='lower right')
    plt.title('Receiver Operating Characteristic')
    plt.plot([(0, 0), (1, 1)], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    # plt.savefig('roc_curve:{}'.format(model_name))


def eval_auc(data):
    TPRandFPR = pd.DataFrame(index=range(len(data)), columns=('TP', 'FP'))
    for j in range(len(data)):
        data1 = data.head(n=j + 1)
        FP = len(data1[data1['truth'] == 0]) / float(len(data[data['truth'] == 0]))
        TP = len(data1[data1['truth'] == 1]) / float(len(data[data['truth'] == 1]))
        TPRandFPR.iloc[j] = [TP, FP]
    AUC= auc(TPRandFPR['FP'],TPRandFPR['TP'])

    # return TPRandFPR
    # return TPRandFPR,AUC
    return AUC


def evaluate(data):
    tn = len(data[(data.truth == 0) & (data.predict == 0)])
    fp = len(data[(data.truth == 0) & (data.predict == 1)])
    fn = len(data[(data.truth == 1) & (data.predict == 0)])
    tp = len(data[(data.truth == 1) & (data.predict == 1)])
    acc = round((tp + tn) / (tp + fp + fn + tn), 3)
    recall = round(tp / (tp + fn), 3)
    precision = round(tp / (tp + fp), 3)
    f1 = precision / (precision + recall)

    return acc, recall, precision, f1


def make_csv(model_name, acc, recall, precision, f1, AUC):
    result = pd.DataFrame({
        'model name': [str(model_name)],
        'Accuracy': [acc],
        'Recall': [recall],
        'Precision': [precision],
        'f1-score': [f1],
        'Area Under Curve': [AUC]
    })

    return result


# 从文件名中获取真实标签
def get_labels(data_path):
    # data_path = 'D:/Download/data/picture'
    total = os.listdir(data_path)
    labels = []
    for sub_path in total:
        file_path = os.path.join(data_path,sub_path)
        if os.path.isdir(file_path):
            name = os.listdir(file_path)[0]
            name = name.split('.')[0] # 保留文件名后缀之前的文件名
            cop = re.compile("[^\u0041-\u005a\u0061-\u007a]") # 正则表达式只保留英文字母 
            name = cop.sub('',name)
            labels.append(name)
    # print(labels)
    return labels