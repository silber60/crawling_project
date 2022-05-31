from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime
import csv
import re
import sys

df_read_csv = pd.read_csv('./예측데이터/wadiz_data_달성률_예측데이터_202205311.csv')
print(df_read_csv)
titles = []
categories = []
rewards = []
rewards_ox = []

for title in df_read_csv['title'] :
    title = re.compile('[^가-힣 ]').sub(' ', title)
    titles.append(title)
df_section_titles = pd.DataFrame(titles, columns=['title'])
for title in df_read_csv['category'] :
    categories.append(title)
df_section_title = pd.DataFrame(categories, columns=['category'])
for title in df_read_csv['reward'] :
    title = re.compile('[^0-9 ]').sub(' ', title)
    rewards.append(title)
sum_o = 0
sum_x = 0
for i in rewards :
    if int(i) >= 1000 :
        rewards_ox.append(1)
        sum_o += 1
    else :
        rewards_ox.append(0)
        sum_x += 1
print(sum_o, sum_x)
df_section_title2 = pd.DataFrame(rewards_ox, columns=['winner'])
df_titles = pd.concat([df_section_titles, df_section_title, df_section_title2], axis=1)
# print(df_titles.head())
# df_titles.info()
print(len(df_titles))
# def title_tag(i):
#     hangul = re.compile('[^ ㄱ-ㅣ가-힣+]') # 한글과 띄어쓰기를 제외한 모든 글자
#     # hangul = re.compile('[^ \u3131-\u3163\uac00-\ud7a3]+')  # 위와 동일
#     titles = hangul.sub('', i) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
#     return(titles)
#
# df_titles = title_tag(df_read_csv)

df_titles.to_csv('./crawling_data(2)/wadiz_new_달성률_예측_데이터_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)





