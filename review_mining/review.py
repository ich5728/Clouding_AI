from os import terminal_size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split

df1 = pd.read_csv('review_mining/data/test_reviews.csv', encoding= 'cp949')
df2 = pd.read_csv('review_mining/data/train_reviews.csv', encoding= 'cp949')

df1.rename(columns = {'별점' : 'ratings'}, inplace = True)
df2.rename(columns = {'별점' : 'ratings'}, inplace = True)
df1.rename(columns = {'리뷰' : 'review'}, inplace = True)
df2.rename(columns = {'리뷰' : 'review'}, inplace = True)

df1['label'] = np.select([df1.ratings > 3], [1], default=0)         # 별점 셋팅(4,5점 = 1, 3,2,1 점 = 0)
df2['label'] = np.select([df2.ratings > 3], [1], default=0)


df1['ratings'].nunique(), df1['review'].nunique(), df1['label'].nunique()       #널 값의 유무 확인
df2['ratings'].nunique(), df2['review'].nunique(), df2['label'].nunique()

print(df2.groupby('label').size().reset_index(name = 'count'))      #비율의 분포 확인

df2['review'] = df2['review'].str.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 ]", "")
df2['review'].replace('', np.nan, inplace=True)
print(df2.isnull().sum())       #한글과 공백을 제외하고 제거 후 빈 샘플이 생기지 않는지 확인

df1.drop_duplicates(subset= ['review'], inplace=True)
df1['review'] = df1['review'].str.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 ]", "")
df1['review'].replace('', np.nan, inplace=True)
df1 = df1.dropna(how='any')
print('전처리 후 테스트용 샘플의 개수 : ', len(df1))

mecab = Mecab()
print(mecab.morphs('와 이런 것도 상품이라고 차라리 내가 만드는 게 나을 뻔'))

