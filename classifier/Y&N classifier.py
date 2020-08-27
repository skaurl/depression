import pandas as pd
import matplotlib.pyplot as plt
import copy
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Reshape, Conv2D, GlobalMaxPooling2D
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

def conv2d_cnn():
    model = Sequential()
    model.add(Embedding(input_dim=2**16, output_dim=output_dim, input_length=max_len))
    model.add(Reshape((max_len, output_dim, 1), input_shape=(max_len, output_dim)))
    model.add(Conv2D(filters=filters, kernel_size=(kernel_size, output_dim), strides=(1, 1), padding='valid'))
    model.add(GlobalMaxPooling2D())

    model.add(Dense(2**6))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

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
    output_dim = 200
    filters = 400
    kernel_size = 5
    epochs = 2**6
    batch_size = 2**10

    depression_dataset_path = r'/gdrive/My Drive/TEST/depression_dataset_맞춤법O.csv'

    depression_dataset = pd.read_csv(depression_dataset_path, encoding='cp949')

    depression_dataset['sentence'] = depression_dataset['sentence'].map(convert_to_ord)
    depression_dataset['label_1'] = LE.fit_transform(depression_dataset['label_1'])

    data = sequence.pad_sequences(depression_dataset['sentence'], maxlen=max_len)
    x_train, x_test, y_train, y_test = train_test_split(data, depression_dataset['label_1'], test_size=0.1, random_state=42)

    print('train_shape : {} / {}'.format(x_train.shape, y_train.shape))
    print('test_shape : {} / {}'.format(x_test.shape, y_test.shape))

    y_true = copy.deepcopy(y_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model, history = conv2d_cnn()

    model.save('/gdrive/My Drive/TEST/model_yn.h5')

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
