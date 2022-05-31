from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime
import csv
import re
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    #scikit-learn 설치
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

df_read_csv = pd.read_csv('./crawling_data(2)/wadiz_new_달성률_예측_데이터_20220531.csv')
training_data = df_read_csv[['title']]
training_data2 = df_read_csv['title']
target = df_read_csv[['winner']]
print(target)
print(len(df_read_csv['title']))
# label_columns = ['category']
# label_df = training_data[label_columns]
# label_df = pd.get_dummies(label_df, columns = label_columns)
# label_df.info()

# preprocessed_data = pd.concat([training_data2, label_df ], axis = 1 )
# print(preprocessed_data.head())
# preprocessed_data.info()

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(
#     preprocessed_data, target, test_size = 0.3)
# print('X_train shape', X_train.shape)
# print('Y_train shape', Y_train.shape)
# print('X_test shape', X_test.shape)
# print('Y_test shape', Y_test.shape)
#
# xy = (X_train, X_test, Y_train, Y_test)
# np.save('./crawling_data(2)/hit.npy', xy)

# X = df['title']
# Y = df['category']


encoder = LabelEncoder()
# labeled_Y = encoder.fit_transform(Y)
# print(labeled_Y[:3])
# label = encoder.classes_
# # print(label)
with open('./models/encoder3.pickle', 'wb') as f:
    pickle.dump(encoder, f)

okt = Okt()
# okt_morph_X = okt.morphs(X[7], stem=True)
# print(okt_morph_X)

for i in range(len(training_data2)):
    training_data2[i] = okt.morphs(training_data2[i], stem=True)
# print(X[:10])

stopwords = pd.read_csv('./stopwords.csv', index_col=0)

for j in range(len(training_data2)):
    words = []
    for i in range(len(training_data2[j])):
        if len(training_data2[j][i]) > 1:
            if training_data2[j][i] not in list(stopwords['stopword']):
                words.append(training_data2[j][i])
    training_data2[j] = ' '.join(words)
print(training_data2[:5])

token = Tokenizer()
token.fit_on_texts(training_data2)
tokened_title = token.texts_to_sequences(training_data2)
print(tokened_title[0])
wordsize = len(token.word_index) + 1
print(wordsize)

with open('./models/news_token3.pickle', 'wb') as f:
    pickle.dump(token, f)

max = 0
for i in range(len(tokened_title)):
    if max < len(tokened_title[i]):
        max = len(tokened_title[i])
print(max)
print(tokened_title)

title_pad = pad_sequences(tokened_title, max)
print(title_pad)

# tokened_title = np.array(title_pad)
# print(tokened_title.shape)
# tokened_title_f = pd.DataFrame(tokened_title, columns=['title'])



X_train, X_test, Y_train, Y_test = train_test_split(title_pad, target, test_size=0.1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save('./models/wadiz_data_winner_max_{}_wordsize_{}'.format(max, wordsize), xy)
