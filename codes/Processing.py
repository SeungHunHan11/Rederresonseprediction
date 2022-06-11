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

class Processing:

    def __init__(self, dataset, colname) :
        self.colname=colname
        self.dataset=pd.DataFrame(dataset)
        self.dataset.rename(columns={self.colname: 'Title'},inplace=True)

    def preprocess(self,Remove_NA=True, Remove_Duplicates=True):
        original_len = len(self.dataset)
        if Remove_NA:
            self.dataset.drop_duplicates(inplace=True)
            print(original_len-len(self.dataset), ' Duplicates removed')
            original_len=len(self.dataset)
        if Remove_Duplicates:
            self.dataset.dropna(inplace=True)
            print(original_len-len(self.dataset), ' Null data removed')
            original_len=len(self.dataset)

    def vectorize(self):

        stopwords = pd.read_csv(
                "https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/korean_stopwords.txt").values.tolist()

        hangul=re.compile('[^ ㄱ-ㅣ 가-힣]')
        result = self.dataset['Title'].apply(lambda x: hangul.sub('',x))
        okt=Okt()
        tokens=result.apply(lambda x: okt.morphs(x))  # Tokenize each sentences

        token_list=[]

        for each_article in tokens:
            removal=[x for x in each_article if x not in stopwords]
            removal=[x for x in each_article if len(x)>1]
            token_list.append(removal)

        self.dataset['Emotion_Ratio']=0
        self.dataset.loc[self.dataset['Bad']==0,'Bad']=0.1 #Dealing with situation when there is no "bad" reaction

        self.dataset['Emotion_Ratio']=self.dataset['Good']/self.dataset['Bad']

        self.dataset.loc[self.dataset['Emotion_Ratio']>=2,'Emotion_Ratio']=2
        self.dataset.loc[(self.dataset['Emotion_Ratio'] < 2) & (self.dataset['Emotion_Ratio'] >=0.5),'Emotion_Ratio']=1
        self.dataset.loc[self.dataset['Emotion_Ratio'] < 0.5,'Emotion_Ratio']=0

        return token_list, self.dataset['Emotion_Ratio']

class Transform:

    def __init__ (self,x):
        self.tokenizer=Tokenizer()
        self.data=x
        self.tokenizer.fit_on_texts(self.data)

    def tokenizing (self, thresholds):
        thresholds=int(thresholds)
        total_cnt = len(self.tokenizer.word_index)
        rare_cnt = 0
        total_freq = 0
        rare_freq = 0

        for key, value in self.tokenizer.word_counts.items():
            total_freq = total_freq + value
            # 단어의 등장 빈도수가 threshold보다 작으면
            if (value < thresholds):
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value

        vocab_size = total_cnt - rare_cnt + 1
        print(round((rare_cnt / total_cnt)*100,2),"% of words are consisted of rare index")
        print('Rare index only appears',round((rare_freq / total_freq)*100,2),'times')
        from tensorflow.keras.preprocessing.text import Tokenizer

        self.tokenizer=Tokenizer(num_words=vocab_size)
        self.tokenizer.fit_on_texts(self.data)
        self.words= self.tokenizer.texts_to_sequences(self.data)

        return self.words,vocab_size

def cleanse(x_list,y_list):

    drop_list = [index for index, sentence in enumerate(x_list) if len(x_list) < 1]
    x_list=np.delete(x_list,drop_list,axis=0).tolist()
    y_list=np.delete(y_list,drop_list,axis=0).tolist()
    y_list=tensorflow.keras.utils.to_categorical(y_list)
    return x_list,y_list

def padding(x_list,pad_size,manuel=True):
    med=statistics.median(map(len, x_list))
    max1=(map(max, x_list))
    print('Median length of all sequences is ',med)
    print('Maximum length of all sequences is ',max1)

    #plt.hist([len(x) for x in x_list], bins=50)
    #plt.xlabel('length of samples')
    #plt.ylabel('number of samples')
    #plt.show()

    if manuel==True:
        pad_size = int(float(input('What is your choice of padding size?: ')))

    x_list2=pad_sequences(x_list,maxlen=pad_size)

    loss=sum(np.subtract(list(map(len,x_list)),list(map(len,x_list2)))>0)

    print(loss, 'cases out of', len(x_list),', ', round(100*loss/len(x_list),2) ,'% of observations have lost some of their elements')

    return x_list2

