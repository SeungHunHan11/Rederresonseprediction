import torch.nn
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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#device=torch.device('cpu')

from Processing import Transform, padding

class Best_model_LSTM(torch.nn.Module):
    def __init__(self,vocab_size,embedding_num,hidden_units,nlayers,nclass,batch=True,drop=0.5):
        super(Best_model_LSTM,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_num=embedding_num
        self.layer1= torch.nn.Embedding(self.vocab_size,
                                        embedding_dim=self.embedding_num,
                                        padding_idx=0)

        self.layer2= torch.nn.LSTM(self.embedding_num,hidden_units,
                                   nlayers,
                                   batch_first=batch,
                                   dropout=drop).to(device)
        self.layer3=torch.nn.Linear(hidden_units, nclass).to(device)

    def forward(self,x):
        xembedded = self.layer1(torch.LongTensor(x))
        xembedded = torch.tensor(xembedded).to(device)
        #xembedded=self.dropout(xembedded)
        lstm_out, (ht, ct)=self.layer2(xembedded)
        scores=self.layer3(ht[-1]).to(device)
        return scores
