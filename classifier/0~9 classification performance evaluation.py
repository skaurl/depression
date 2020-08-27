import pandas as pd
import matplotlib.pyplot as plt
import copy
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder; LE = LabelEncoder()
from sklearn.metrics import classification_report
import scikitplot as skplt

def convert_to_ord(data):
    try:
        return [ord(xx) for xx in data]
    except:
        print(data)

if __name__ == "__main__":
    max_len = 64

    depression_dataset_path = r'/Users/kimnamhyeok/Google 드라이브/한양대학교/2020 프로젝트 학기제/자료/data/dataset/all/depression_dataset_맞춤법O.csv'

    depression_dataset = pd.read_csv(depression_dataset_path, encoding='cp949')
    depression_dataset = depression_dataset[depression_dataset['label_1']=='Y']

    depression_dataset['sentence'] = depression_dataset['sentence'].map(convert_to_ord)
    depression_dataset['label_2'] = LE.fit_transform(depression_dataset['label_2'])

    data = sequence.pad_sequences(depression_dataset['sentence'], maxlen=max_len)
    x_train, x_test, y_train, y_test = train_test_split(data, depression_dataset['label_2'], test_size=0.1, random_state=42)

    print('train_shape : {} / {}'.format(x_train.shape, y_train.shape))
    print('test_shape : {} / {}'.format(x_test.shape, y_test.shape))

    y_true = copy.deepcopy(y_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = load_model('/Users/kimnamhyeok/Google 드라이브/한양대학교/2020 프로젝트 학기제/자료/data/classifier/model_09.h5')

    y_true = list(y_true)
    y_pred = model.predict_classes(x_test)
    y_pred = list(y_pred)

    print(classification_report(y_true, y_pred))
    print(pd.crosstab(pd.Series(y_true), pd.Series(y_pred), rownames=['True'], colnames=['Predicted']))

    plt.rcParams["figure.figsize"] = (7, 7)
    skplt.metrics.plot_roc(y_true, model.predict_proba(x_test))
    plt.xlim(-0.025, 1.025)
    plt.ylim(-0.025, 1.025)
    plt.show()
