import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset.csv', encoding='cp949')
df = df.drop_duplicates()

print('문장의 개수 :', str(len(df))+'문장')
print('문장의 최대 길이 :', str(max(len(l) for l in df['sentence']))+'글자')
print('문장의 평균 길이 :', str(round(sum(map(len, df['sentence'])) / len(df['sentence'])))+'글자')
#데이터셋의 기본 통계

plt.hist([len(s) for s in df['sentence']], bins=20)
plt.xlim(0,200)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
plt.show()
#데이터셋의 문장 길이별 빈도수 그래프

sns.kdeplot([len(s) for s in df['sentence']], shade=True)
plt.title('Kernel Density Estimation')
plt.show()
#데이터셋의 커널 밀도 추정

print(df.groupby('label_1').count()['sentence'])
df.groupby('label_1').count()['sentence'].plot(kind = 'bar')
plt.show()
#label_1의 빈도수 그래프

print(df.groupby(['label_1', 'label_2']).count()['sentence'])
df.groupby(['label_1', 'label_2']).count()['sentence'].plot(kind = 'bar')
plt.show()
#label_2의 빈도수 그래프

print(df.describe())
#데이터셋의 요약 통계