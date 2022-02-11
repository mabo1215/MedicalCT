#--coding:utf-8-- Bob

from typing import NamedTuple

import csv
from typing import NamedTuple
import os
import json
import argparse
import logging
import sys

import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import torch
import re
import torch.nn.functional as Fnn
from torch.utils.data import TensorDataset, DataLoader
from bobbert import DistilBertTokenizer,  DistilBertForSequenceClassification
# from bobbert import BertTokenizer,  BertForSequenceClassification
# from kfp.v2.dsl import (
#     component,
#     Input,
#     Output,
#     Artifact,
#     Model,
#     Metrics,
#     ClassificationMetrics,
# )


# @component(
#     packages_to_install=["sklearn"],
#     base_image='gcr.io/l153711648525780/item-train-gpu-with-ts',
# )
def evaluate(
    input_model: str,
    max_seq_len: int,
    test_data_dir: int,
    vocab_dir: int,
    metrics_out1: str,
    metrics_out2: str,
    pretrained_model_name: str,
) -> NamedTuple("Outputs", [("accuracy", float)]):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    test_data_input_root = test_data_dir
    test_df = pd.read_csv(f'{test_data_input_root}/batchtest.csv',encoding='utf-8')

    index_df = pd.read_json(f'{test_data_input_root}/index_to_name.json', orient='index')
    indexes = index_df[0].to_list()

    # pretrained_model_name =  "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name)

    test_encodings = tokenizer.batch_encode_plus(
        test_df['text'].values,
        add_special_tokens=True,
        padding='max_length',
        max_length=max_seq_len,
        truncation=True,
    )

    test_dataset = TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask']),
        torch.tensor(test_df['label'].values),
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained(input_model).to(device)
    model.eval()

    y_pred = []
    y_true = []
    pred_res = []
    with torch.no_grad():
        for text, attention_mask, label in tqdm.tqdm(test_loader):
            labels = label.type(torch.long).to(device)
            text = text.type(torch.LongTensor).to(device)
            attention_mask = attention_mask.type(torch.LongTensor).to(device)

            outputs = model(text, attention_mask=attention_mask, labels=labels)

            pred_res.append(torch.argmax(outputs[1],dim=1))
            y_pred += torch.argmax(outputs[1], dim=1).tolist()
            y_true += labels.tolist()
    # y_pred_whole = []
    # for ctrs in pred_res:
    #     csry = ctrs.tolist()
    #     # print(len(csry),csry)
    #     for t in range(len(csry)):
    #         y_pred_whole.append(csry[t])
    # if len(y_pred_whole) == len(y_true):
    #     for i in range(len(y_true)):
    #         print(y_true[i]," ",y_pred_whole[i],'\n')
    report = classification_report(y_true, y_pred, output_dict=True)
    # logger.info('Classification report: \n{}'.format(report))
    # for i in report:
    #     if i <'9' and i > '0':
    #     # rname1 = indexes[int(i[0])]
    #         print(i,indexes[int(i)])
    df = pd.DataFrame(report).transpose()
    df.to_csv(metrics_out1,index=True)


    # logger.info('indexes: {}'.format(indexes))
    # s = 0
    # # with open("E:/source/JointBERT/run/rawsets/expdistilsub/indextoname.csv", 'w', newline='') as csvfile3:
    # #     spamwriter = csv.writer(csvfile3)
    # for i in indexes:
    #     # spamwriter.writerow([s, i])
    #     print(s,i)
    #     s+=1
    # confumx = confusion_matrix(y_true, y_pred).tolist()
    # logger.info('confusion matrix: \n{}'.format(confumx))
    with open(metrics_out2, 'w', newline='') as csvfile2:
        spamwriter = csv.writer(csvfile2)
        spamwriter.writerow(["Index","Predict title", "Predict Result", "True Label"])
        if len(y_pred) == len(y_true):
            for i in tqdm.tqdm(range(len(y_pred))):
                pre_idx = int(y_pred[i])
                tre_idx = int(y_true[i])
                tes_restr = re.sub(u"([^\u0020\0024\u0025\u0030-\u0039\u0040-\u005a\u0061-\u007a])", " ", test_df['text'][i])
                if pre_idx != tre_idx:
                    spamwriter.writerow([i,tes_restr,indexes[pre_idx],indexes[tre_idx]])
                # pred_res[pre_t]
    csvfile2.close()
    # df2 = pd.DataFrame(confumx).transpose()
    # df2.to_csv(metrics_out2,index=True)

    logger.info('Validation accuracy: \n{}'.format(report['accuracy']))
    return (report['accuracy'], )

def change_report(metrics1,test_data_dir):
    report_df = pd.read_csv(f'{metrics1}', encoding='utf-8')
    index_df = pd.read_json(f'{test_data_dir}/index_to_name.json', orient='index')
    indexes = index_df[0].to_list()
    with open(f'{test_data_dir}/tmp.csv', 'w', newline='') as csvfile2:
        spamwriter = csv.writer(csvfile2)
        spamwriter.writerow(["index", "precision", "recall","f1 - score","support"])
        for i in range(len(report_df['index'])):
            s = report_df['index'].values[i]
            if (s).isdigit():
                spamwriter.writerow([indexes[int(s)], report_df["precision"].values[i], report_df["recall"].values[i], report_df["f1-score"].values[i],report_df["support"].values[i]])
                print(indexes[int(s)])
            else:
                spamwriter.writerow([report_df["index"].values[i], report_df["precision"].values[i], report_df["recall"].values[i], report_df["f1-score"].values[i],report_df["support"].values[i]])
    csvfile2.close()
    os.remove(metrics1)
    os.rename(f'{test_data_dir}/tmp.csv', metrics1)


if __name__ == '__main__':
    # print(f'Starting {__file__} with environment: {os.environ}')

    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    parser.add_argument('--data-dir', type=str, default="E:/source/JointBERT/run/rawsets/expdistilsub/")
    parser.add_argument('--vocab-dir', type=str, default="E:/source/JointBERT/models/distilbert-base-uncased/")
    parser.add_argument('--model-dir', type=str,default="run/train/expdistilsub/")
    parser.add_argument('--max-seq-len', type=int, default=200)
    parser.add_argument('--met', type=str, default="./res/res.csv")
    parser.add_argument('--met2', type=str,default="./res/batinfres.csv")
    parser.add_argument('--pre-train-name', type=str,default="distilbert-base-uncased")
    # pretrained_model_name =  "distilbert-base-uncased"

    args = parser.parse_args()

    pretrained_model_name = args.pre_train_name
    test_data_dir = args.data_dir
    input_model = args.model_dir
    max_seq_len = args.max_seq_len
    metrics1, metrics2 = args.met, args.met2
    vocab_dir = args.vocab_dir
    resacc = evaluate(input_model,max_seq_len,test_data_dir,vocab_dir ,metrics1,metrics2,pretrained_model_name)
    print(resacc)
    # change_report(metrics1,test_data_dir)