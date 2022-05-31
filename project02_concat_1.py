import pandas as pd
import glob
import datetime

# data_path = glob.glob('./crawling_data/*')
data_path = glob.glob('./예측데이터/*')
print(data_path)

df = pd.DataFrame()
for path in data_path[1:] :
    df_temp = pd.read_csv(path) #임시 저장
    df = pd.concat([df, df_temp])
df.dropna(inplace=True) #None 값 제거
df.reset_index(inplace=True,drop = True) #제거했으니 reset #인덱스 있는거 합칠 땐 drop = True
print(df.head())
print(df.tail())
df.info()

df.to_csv('./예측데이터/wadiz_data_달성률_예측데이터_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)
# df.to_excel('./crawling_data/wadiz_data_{}.xlsx'.format(
#     datetime.datetime.now().strftime('%Y%m%d')), index=False)