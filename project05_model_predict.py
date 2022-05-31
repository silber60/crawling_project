# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU를 쓰도록 강제
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    #scikit-learn 설치
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model


pd.set_option('display.unicode.east_asian_width',True)
pd.set_option('display.max_columns', 20)
df = pd.read_csv('./crawling_data(2)/wadiz_new_달성률_예측_데이터_20220531.csv')
print(df.head())
df.info()
# exit()
X = df['title']
Y = df['winner']


# encoder = LabelEncoder()
# labeled_Y = encoder.transform(Y)
# print(labeled_Y[:3])
# label = encoder.classes_
# print(label)

# with open('./models/encoder3.pickle', 'rb') as f:
#     encoder = pickle.load(f)
# label = encoder.classes_
# print(label)
# onehot_Y = to_categorical(labeled_Y)
# print(onehot_Y)

okt = Okt()
# okt_morph_X = okt.morphs(X[7], stem=True)
# print(okt_morph_X)
#okt.morphs() 텍스트를 형태소 단위로 나눈다. 옵션으로 norm과 stem이 있다. norm은 문장을 정규화. stem은 각 단어에서 어간을 추출.(기본값은 둘다 False)
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
# print(X[:10])

stopwords = pd.read_csv('./stopwords.csv', index_col=0)

for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
# print(X[:5])

with open('./models/news_token2.pickle', 'rb') as f:
    token = pickle.load(f)

tokened_X = token.texts_to_sequences(X)
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 13:
        tokened_X[i] = tokened_X[i][:13]

X_pad = pad_sequences(tokened_X, 13)
print((X_pad[:5]))

model = load_model('./models/wadiz_classfication_model_winner_0.7690253853797913.h5')
preds = model.predict(X_pad)
print(preds)

predicts = []
for pred in preds:
    predicts.append(np.round(pred, 0))
df['predict'] = predicts

print(df.head(30))

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'winner'] == df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 'O'
    else:
        df.loc[i, 'OX'] = 'X'
print(df.head(30))

print(df['OX'].value_counts())
print(df['OX'].value_counts()/len(df))

for i in range(len(df)):
    if df['winner'][i] != df['predict'][i]:
        print(df.iloc[i])





