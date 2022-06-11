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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def embed(x,size,):

    return x

class LSTM(torch.nn.Module):
    def __init__(self,embeddingdim,hiddensize,num_layers,droprate,n_class,batch=True):
        super(LSTM,self).__init__()
        self.hiddensize=hiddensize
        self.num_layers = num_layers
        self.lstm=torch.nn.LSTM(embeddingdim, hiddensize, num_layers, batch_first=batch,dropout=droprate).to(device)
        self.fc = torch.nn.Linear(hiddensize, n_class).to(device)

    def forward(self, x , isembedding=False):
        if isembedding==True:
            x = self.embedding(torch.LongTensor(x))
            x=torch.tensor(x)
            x=x.to(device)

        lstm_out, (ht, ct) = self.lstm(x)
        score= self.fc(ht[-1])
        return score

