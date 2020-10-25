'''
    Copyright (c) 2018-2020
    Jianjia Ma
    majianjia@live.com
    SPDX-License-Identifier: Apache-2.0
    Change Logs:
    Date           Author       Notes
    2019-06-30     Jianjia Ma   The first version
'''

import sys
import os
sys.path.append(os.path.abspath("../../scripts"))
print(sys.path)

from tensorflow.keras import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import load_model, save_model
import tensorflow as tf
import numpy as np
from nnom import *

save_dir = 'keras_mnist_trained_model.h5'

def build_model(x_shape):
    inputs = Input(shape=x_shape)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid')(inputs)
    x = BatchNormalization()(x)

    x = Conv2D(16, dilation_rate=(1,1), kernel_size=(5, 5), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding="same")(x)
    x = Dropout(0.2)(x)

    x = DepthwiseConv2D(depth_multiplier=2, dilation_rate=(2,2), kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU(negative_slope=0.2, threshold=0, max_value=6)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding="same")(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(10)(x)
    # x = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    # x = GlobalAveragePooling2D()(x)
    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model

def train(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=50):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              shuffle=False)

    save_model(model, save_dir)
    del model
    tf.keras.backend.clear_session()
    return history

def main():
    #physical_devices = tf.config.experimental.list_physical_devices("GPU")
    #if(physical_devices is not None):
    #    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    epochs = 2
    num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test_original, y_test_original) = mnist.load_data()

    x_test = x_test_original
    y_test = y_test_original
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # reshape to 4 d becaue we build for 4d?
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    print('x_train shape:', x_train.shape)

    # quantize the range to q7
    x_test = x_test.astype('float32')/255
    x_train = x_train.astype('float32')/255
    print("data range", x_test.min(), x_test.max())

    #build model
    model = build_model(x_test.shape[1:])

    # train model
    history = train(model, x_train, y_train, x_test.copy(), y_test.copy(), epochs=epochs)

    # -------- generate weights.h (NNoM model) ----------
    # get the best model
    model = load_model(save_dir)

    # output test
    # for L in model.layers:
    #     layer_model = Model(inputs=model.input, outputs=L.output)
    #     print("layer", L.name)
    #     features = layer_model.predict(x_test[:1])
    #     pass
    #     #print("output", features)

    # only use 1000 for test
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    # generate binary dataset for NNoM validation, 0~1 -> 0~127, q7
    generate_test_bin(x_test*127, y_test, name='test_data.bin')

    # evaluate in Keras (for comparision)
    scores = evaluate_model(model, x_test, y_test)

    # generate NNoM model, x_test is the calibration dataset used in quantisation process
    generate_model(model,  x_test, format='hwc', per_channel_quant=False, name="weights.h")

    # --------- for test in CI ----------
    # build NNoM
    os.system("scons")

    # do inference using NNoM
    cmd = ".\mnist.exe" if 'win' in sys.platform else "./mnist"
    os.system(cmd)
    try:
        # get NNoM results
        result = np.genfromtxt('result.csv', delimiter=',', dtype=np.int, skip_header=1)
        result = result[:,0]        # the first column is the label, the second is the probability
        label = y_test_original[:len(y_test)].flatten()     # use the original numerical label
        acc = np.sum(result == label).astype('float32')/len(result)
        if (acc > 0.5):
            print("Top 1 Accuracy on Keras %.2f%%" %(scores[1]*100))
            print("Top 1 Accuracy on NNoM  %.2f%%" %(acc *100))
            return 0
        else:
            raise Exception('test failed, accuracy is %.1f%% < 80%%' % (acc * 100.0))
    except:
        raise Exception('could not perform the test with NNoM')

if __name__ == "__main__":
    main()
