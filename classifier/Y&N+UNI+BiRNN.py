import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Activation, Bidirectional
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder; LE = LabelEncoder()
from sklearn.metrics import classification_report

def convert_to_ord(data):
    try:
        return [ord(xx) for xx in data]
    except:
        print(data)

def birnn():
    model = Sequential()
    model.add(Bidirectional(SimpleRNN(2 ** 3, return_sequences=True), input_shape=(max_len, 1)))
    model.add(Bidirectional(SimpleRNN(2 ** 3, return_sequences=True)))
    model.add(Bidirectional(SimpleRNN(2 ** 3, return_sequences=False)))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    early_stopping = EarlyStopping(patience=10)
    history = model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])

    return model, history

if __name__ == "__main__":
    max_len = 64
    epochs = 2**7
    batch_size = 2**10

    depression_dataset_path = r'depression_dataset_맞춤법O.csv'

    depression_dataset = pd.read_csv(depression_dataset_path, encoding='cp949')

    depression_dataset['sentence'] = depression_dataset['sentence'].map(convert_to_ord)
    depression_dataset['label_1'] = LE.fit_transform(depression_dataset['label_1'])

    data = sequence.pad_sequences(depression_dataset['sentence'], maxlen=max_len)
    x_train, x_test, y_train, y_test = train_test_split(data, depression_dataset['label_1'], test_size=0.1, random_state=42)

    x_train = np.array(x_train).reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = np.array(x_test).reshape((x_test.shape[0], x_test.shape[1], 1))

    y_true = copy.deepcopy(y_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('train_shape : {} / {}'.format(x_train.shape, y_train.shape))
    print('test_shape : {} / {}'.format(x_test.shape, y_test.shape))

    model, history = birnn()

    scores = model.evaluate(x_test, y_test)
    print(scores)
    print("정확도: %.2f%%" % (scores[1] * 100))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.figure()
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    y_true = list(y_true)
    y_pred = model.predict_classes(x_test)
    y_pred = list(y_pred)

    print(classification_report(y_true, y_pred))
    print(pd.crosstab(pd.Series(y_true), pd.Series(y_pred), rownames=['True'], colnames=['Predicted']))
