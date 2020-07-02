import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import kss
import matplotlib.pyplot as plt

df = pd.read_csv('naver_blog.csv',encoding='cp949')
#네이버 블로그 크롤링 데이터 불러오기

A = []
for i in tqdm(range(len(df))):
    df.iloc[i,2] = ' '.join(re.sub(r'[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', ' ', str(df.iloc[i,2].strip())).split())
    #제목에서 숫자, 영어, 한글만 추출

    B = kss.split_sentences(df.iloc[i,3])
    #kss라이브러리를 이용하여 문장 단위로 한번 더 분리

    for j in range(len(B)):
        B[j] = ' '.join(re.sub(r'[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', ' ', str(B[j].strip())).split())
        #문장에서 숫자, 영어, 한글만 추출

        if len(B[j]) >= 6 and len(B[j]) < 200:
            #문장의 길이가 6 이상 200 이하의 문장만 추출
            A.append({'id': df.iloc[i, 0], 'url': df.iloc[i, 1], 'title': df.iloc[i, 2], 'content': B[j], 'date': df.iloc[i, 4]})

df = pd.DataFrame(A)
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

df.to_csv('naver_blog.csv', index=False, encoding='cp949')

print('문장의 개수 :', len(df))
print('문장의 최대 길이 :', max(len(l) for l in df['content']))
print('문장의 평균 길이 :', sum(map(len, df['content'])) / len(df['content']))
#네이버 블로그 데이터의 기본 통계

plt.hist([len(s) for s in df['content']], bins=100)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
plt.show()
#네이버 블로그 데이터의 문장 길이별 빈도수 그래프