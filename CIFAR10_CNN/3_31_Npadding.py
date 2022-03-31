import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

from tensorflow.keras.datasets import cifar10
from tensorflow.python.client import device_lib

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device_lib.list_local_devices()

    NUM_CLASSES = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    input_layer = Input((32, 32, 3))

    x = Conv2D(filters=10, kernel_size=3, strides=1, activation = 'relu')(input_layer)

    x = Conv2D(filters=30, kernel_size=3, strides=1,  activation = 'relu')(x)

    x = Conv2D(filters=100, kernel_size=3, strides=1,  activation = 'relu')(x)


    x = Conv2D(filters=200, kernel_size=3, strides=1,  activation = 'relu')(x)


    x = Flatten()(x)

    x = Dense(128, activation = 'relu')(x)


    x = Dense(NUM_CLASSES)(x)
    output_layer = Activation('softmax')(x)

    model = Model(input_layer, output_layer)
    model.summary()

    opt = Adam(lr=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(x_train
              , y_train
              , batch_size=32
              , epochs=50
              , shuffle=True
              , validation_data=(x_test, y_test))

    model.evaluate(x_test, y_test, batch_size=1000)