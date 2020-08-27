import pandas as pd
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder; LE = LabelEncoder()
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def convert_to_ord(data):
    try:
        return [ord(xx) for xx in data]
    except:
        print(data)

max_len = 64

depression_dataset_path = r'depression_dataset_맞춤법O.csv'

depression_dataset = pd.read_csv(depression_dataset_path, encoding='cp949')
depression_dataset = depression_dataset[depression_dataset['label_1']=='Y']

depression_dataset['sentence'] = depression_dataset['sentence'].map(convert_to_ord)
depression_dataset['label_2'] = LE.fit_transform(depression_dataset['label_2'])

data = sequence.pad_sequences(depression_dataset['sentence'], maxlen=max_len)
x_train, x_test, y_train, y_test = train_test_split(data, depression_dataset['label_2'], test_size=0.1, random_state=42)

print('train_shape : {} / {}'.format(x_train.shape, y_train.shape))
print('test_shape : {} / {}'.format(x_test.shape, y_test.shape))

model = GaussianNB()
model.fit(x_train, y_train)

y_test = list(y_test)
y_pred = list(model.predict(x_test))

print('accuracy_score = ', accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
print(pd.crosstab(pd.Series(y_test), pd.Series(y_pred), rownames=['True'], colnames=['Predicted']))