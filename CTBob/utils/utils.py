import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve , confusion_matrix
import warnings
import itertools

warnings.filterwarnings("ignore")


def deal_csv(csv_url, ids, labels):
    data = pd.read_csv(csv_url)
    data.columns = ['truth' , 'predict']
    for i in range(0,len(labels)):
        df_mask2 = data['predict'].str.contains(labels[i])
        for mask_idx in range(len(df_mask2)):
            if df_mask2[mask_idx] == True:
                data.loc[mask_idx,'predict'] = i
    for i in range(0,len(ids)):
        df_mask = data['truth'].str.contains(ids[i])
        for mask_idx in range(0,len(df_mask)):
            if df_mask[mask_idx] == True:
                data.loc[mask_idx,'truth'] = i
    # data['truth'] = data['truth'].values.contains(ids).index()
    # data['predict'] = data['predict'].apply(lambda x: x if labels[x] in x else 0)
    # data['truth'] = data['truth'].apply(lambda x: x if ids[x] in x else 0)
    # data['predict'] = data['predict'].apply(lambda x: x if labels[x] in x else 0)
    # data.sort_values('predict', inplace=True, ascending=False)
    return data


def plot_cm(data):
    y_truth = data['truth'].values
    y_predict = data['predict'].values
    cm = confusion_matrix(y_truth.tolist(),y_predict.tolist())
    # cm = np.arange(4).reshape(2, 2)
    # cm[0, 0] = len(data[(data.truth == 0) & (data.predict == 0)])
    # cm[0, 1] = len(data[(data.truth == 0) & (data.predict == 1)])
    # cm[1, 0] = len(data[(data.truth == 1) & (data.predict == 0)])
    # cm[1, 1] = len(data[(data.truth == 1) & (data.predict == 1)])
    classes = [0, 1, 2,3]
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


def plot_roc(data,model_name):
    # data.columns = ['truth' , 'predict']
    y_truth = data['truth'].values
    y_predict = data['predict'].values
    fpr,tpr,thresholds = roc_curve(y_truth.tolist(),y_predict.tolist(),pos_label=2)
    roc_auc = auc(fpr,tpr)
    # data.sort_values('predict', inplace=True, ascending=False)
    # TPRandFPR = pd.DataFrame(index=range(len(data)), columns=('TP', 'FP'))
    # for j in range(len(data)):
    #     data1 = data.head(n=j + 1)
    #     FP = len(data1[data1['truth'] == 0]) / float(len(data[data['truth'] == 0]))
    #     TP = len(data1[data1['truth'] == 1]) / float(len(data[data['truth'] == 1]))
    #     TPRandFPR.iloc[j] = [TP, FP]
    # AUC = auc(TPRandFPR['FP'], TPRandFPR['TP'])
    # # plt.scatter(x=TPRandFPR['FP'],y=TPRandFPR['TP'],label='(FPR,TPR)',color='b')
    # plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], 'k', label='AUC = %0.2f' % AUC)
    plt.plot(fpr, tpr, 'k', label='AUC = %0.2f' % roc_auc, lw =2)
    plt.legend(loc='lower right')
    plt.title('Receiver Operating Characteristic')
    plt.plot([(0, 0), (1, 1)], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig('roc_curve:{}'.format(model_name))


def eval_roc(data):
    TPRandFPR = pd.DataFrame(index=range(len(data)), columns=('TP', 'FP'))
    for j in range(len(data)):
        data1 = data.head(n=j + 1)
        FP = len(data1[data1['truth'] == 0]) / float(len(data[data['truth'] == 0]))
        TP = len(data1[data1['truth'] == 1]) / float(len(data[data['truth'] == 1]))
        TPRandFPR.iloc[j] = [TP, FP]
    AUC= auc(TPRandFPR['FP'],TPRandFPR['TP'])

    # return TPRandFPR
    return TPRandFPR,AUC


def evaluate_res(data):
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
