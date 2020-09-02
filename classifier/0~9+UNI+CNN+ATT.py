import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Embedding, concatenate, BatchNormalization, Dropout, Conv2D, MaxPool2D, \
    Reshape, Attention
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder;
LE = LabelEncoder()

depression_dataset_path = r'depression_dataset_맞춤법O.csv'

depression_dataset = pd.read_csv(depression_dataset_path, encoding='cp949')
depression_dataset = depression_dataset[depression_dataset['label_1'] != 'N']
depression_dataset = depression_dataset[depression_dataset['label_2'] != '?']


def convert_to_ord(data):
    try:
        return [ord(xx) for xx in data]
    except:
        print(data)

class Self_Attention_Layer(Layer):
    def __init__(self, output_dim, **kwargs):
        super(Self_Attention_Layer, self).__init__(**kwargs)

        self.wq = Dense(output_dim)
        self.wk = Dense(output_dim)
        self.wv = Dense(output_dim)
        self.out = Dense(output_dim)

    def self_attention_(self, q, k, v):
        attention_map_ = tf.matmul(q, k, transpose_b=True)
        attention_weights_ = tf.nn.softmax(attention_map_, axis=-1)
        output_ = tf.matmul(attention_weights_, v)

        return output_, attention_weights_

    def call(self, Q, K, V):
        q = self.wq(Q)
        k = self.wk(K)
        v = self.wv(V)

        self_attention, attention_wiehgts = self.self_attention_(q, k, v)
        output = self.out(self_attention)

        return output, attention_wiehgts

def create_model(char_x_train, y_train_data, maxlen):
    # parameters
    global emd_size
    global output_size

    # model
    filter_sizes = (2, 3, 4, 5)
    filter_num = 400
    text_input = Input(shape=(maxlen,), name="char_input", dtype="float32")
    embbed_text = Embedding(2 ** 16, emd_size, name="embed_char")(text_input)
    # ex_emd_text = Reshape((maxlen, emd_size, 1))(embbed_text)
    convs = []
    for filter_size in filter_sizes:
        # convolution layer
        conv = Conv1D(filter_num, filter_size, activation="relu", padding="same")(embbed_text)
        # conv = Conv2D(filter_num, (filter_size, emd_size), activation="relu",strides=(1,1), padding = "same")(ex_emd_text)
        self_attention_conv, weights = Self_Attention_Layer(400)(conv, conv, conv)
        max_pool = GlobalMaxPooling1D()(self_attention_conv)
        # max_pool = MaxPool2D()(self_attention_conv)
        # attention layer
        convs.append(max_pool)
    convs_merged = concatenate(convs)
    fc_Layer = Dense(2**7)(convs_merged)
    dropout_Layer = Dropout(0.2)(fc_Layer)
    Batch_Layer = BatchNormalization()(dropout_Layer)
    output_Layer = Dense(output_size, activation="softmax")(Batch_Layer)
    model = Model(text_input, output_Layer)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    early_stopping = EarlyStopping(patience=10)
    history = model.fit(char_x_train, y_train_data, epochs=2 ** 10, batch_size=2 ** 10, validation_split=0.1, shuffle=True, callbacks=[early_stopping])

    return model, history

if __name__ == "__main__":
    # global
    max_length = 64
    emd_size = 200
    output_size = 10

    depression_dataset['sentence'] = depression_dataset['sentence'].map(convert_to_ord)
    depression_dataset['label_2'] = LE.fit_transform(depression_dataset['label_2'])

    char_train_data = sequence.pad_sequences(depression_dataset['sentence'], maxlen=max_length)
    x_train, x_test, y_train, y_test = train_test_split(char_train_data, depression_dataset['label_2'], random_state=42)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    create_model, history = create_model(x_train, y_train, max_length)

    scores = create_model.evaluate(x_test, y_test)
    print(scores)
    print("정확도: %.2f%%" % (scores[1] * 100))