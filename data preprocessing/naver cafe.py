import pickle
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

with open('naver_cafe.pickle', 'rb') as f:
    df = pickle.load(f)
    #네이버 카페 크롤링 데이터 불러오기

for i in tqdm(range(len(df))):
    df.iloc[i,2] = ' '.join(re.sub(r'[^a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', ' ', str(df.iloc[i,2])).split())
    #제목에서 숫자, 영어, 한글만 추출
    df.iloc[i,3] = ' '.join(re.sub(r'[^a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', ' ', str(df.iloc[i,3])).split())
    #문장에서 숫자, 영어, 한글만 추출

df[df['content']==''] = np.nan
#빈 데이터를 nan으로 변경

df = df.dropna(how = 'any')
#nan이 있는 행을 제거

df = df.drop_duplicates()
#중복 제거

df = df.reset_index()
del df['index']
#index 재정렬

df['date'] = pd.to_datetime(df['date'])
#date 열을 datetime형으로 변경

df.to_csv('naver_cafe.csv', index=False, encoding='cp949')

df = pd.read_csv('naver_cafe.csv', encoding='cp949')

print('게시 글의 개수 :', len(df))
print('게시 글의 최대 길이 :', max(len(l) for l in df['content']))
print('게시 글의 평균 길이 :', sum(map(len, df['content'])) / len(df['content']))
#네이버 카페 데이터의 기본 통계

plt.hist([len(s) for s in df['content']], bins=100)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
plt.show()
#네이버 카페 데이터의 문장 길이별 빈도수 그래프