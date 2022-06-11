from sklearn.model_selection import KFold
import tensorflow.keras.datasets.imdb
import torch
import numpy as np
import pandas as pd
import re
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import statistics
import sys
from torch import LongTensor
import random
import os
import Models
from Models import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(1998)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def LSTM_Best(vocab_size,embedding_opt,hidden_opt,layers_opt, x,y,decay=1e-5,nfolds=5,nepochs=10,batch_size=19,drops=0.4,saving=False):



    loss_train_all = []
    loss_val_all = []
    acc_train_all = []
    acc_val_all = []
    best_acc = 0
    embed_dim = []
    hidden_u = []
    lay = []

    split = KFold(n_splits=nfolds, shuffle=True, random_state=True)
    train_loss_record = 0
    val_loss_record = 0
    train_acc_record = 0
    val_acc_record = 0
    keeptrack = 0

    criterion = torch.nn.CrossEntropyLoss() # 분류


    for embedding_num in embedding_opt:
        model = torch.nn.Sequential()

        embedding = torch.nn.Embedding(vocab_size, embedding_dim=embedding_num, padding_idx=0)
        xembedded = embedding(torch.LongTensor(x))
        xembedded = torch.tensor(xembedded).to(device)
        dataset_train = TensorDataset(xembedded, torch.tensor(y))
        trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        x1 = torch.zeros((len(trainloader) - 1, batch_size, 11, embedding_num))
        y1 = torch.zeros((len(trainloader) - 1, batch_size, 3))

        idx = 0

        for xx, yy in trainloader:
            try:
                x1[idx] = xx
                y1[idx] = yy
            except:
                continue
            idx += 1

        train_x_set, val_x_set, train_y_set, val_y_set = train_test_split(x1, y1, test_size=0.3, shuffle=True)

        for hidden in hidden_opt:
            for layers in layers_opt:
                keeptrack += 1
                embed_dim.append(embedding)
                hidden_u.append(hidden)
                lay.append(layers)
                loss_train = []
                loss_val = []
                acc_train = []
                acc_val = []

                lstm = LSTM(embedding_num, hidden, num_layers=layers, n_class=3, batch=True,droprate=drops).to(device)
                optimizer = torch.optim.Adam(lstm.parameters(), lr=0.05,weight_decay= decay)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                                        last_epoch=-1,
                                                        verbose=False)

                for epochs in tqdm(range(nepochs)):
                    train_loss_record = 0
                    val_loss_record = 0
                    train_acc_record = 0
                    val_acc_record = 0

                    for train_idx, val_idx in split.split(train_x_set, train_y_set):
                        train_x, train_y = train_x_set[train_idx], train_y_set[train_idx]
                        val_x, val_y = train_x_set[val_idx], train_y_set[val_idx]

                        train_x, val_x = train_x.view(-1,11,embedding_num).to(device), val_x.view(-1,11,embedding_num).to(device)
                        train_y, val_y = train_y.view(-1,3).to(device), val_y.view(-1,3).to(device)

                        lstm.train()

                        scores = lstm(train_x)
                        loss = criterion(scores, train_y)
                        train_loss_record += (loss.item())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        train_acc_record += (sum(
                            torch.argmax(train_y, axis=1) == torch.argmax(scores, axis=1)) / len(train_y)).item()

                        lstm.eval()

                        with torch.no_grad():
                            scores2 = lstm(val_x)
                            val_loss_record += criterion(scores2, val_y).item()
                            val_acc_record += (sum(
                                torch.argmax(val_y, axis=1) == torch.argmax(scores2, axis=1)) / len(val_y)).item()

                    acc_val.append(val_acc_record / nfolds)
                    loss_val.append(val_loss_record / nfolds)

                    loss_train.append(train_loss_record / nfolds)
                    acc_train.append(train_acc_record / nfolds)

                    if saving:
                        if best_acc < max(acc_val):
                            best_acc = max(acc_val)
                            print(best_acc)
                            torch.save(lstm.state_dict(),
                                       'C:/Users/icako/NewsCrawler/best_model.pth')
                            print('Model Saved.')

                print('Maximum Validation accuracy: ', max(acc_val))
                print('Minimum Validation loss: ', min(loss_val))
                loss_train_all.append(min(loss_train))
                loss_val_all.append(min(loss_val))
                acc_train_all.append(max(acc_train))
                acc_val_all.append(max(acc_val))

                print(keeptrack, ' Combinations out of 27 Done')



    dic={'embedding_opt':embed_dim,'hidden_units':hidden_u,"n_layers":lay,
         'loss_train':loss_train_all,'loss_val':loss_val_all,
         'acc_train':acc_train_all,'acc_val':acc_val_all}

    return dic, val_x_set, val_y_set