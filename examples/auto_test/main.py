'''
    Copyright (c) 2018-2019
    Jianjia Ma, Wearable Bio-Robotics Group (WBR)
    majianjia@live.com
    SPDX-License-Identifier: Apache-2.0
    Change Logs:
    Date           Author       Notes
    2019-06-30     Jianjia Ma   The first version
'''

#import matplotlib.pyplot as plt
import sys
import os
os.sys.path.append(os.path.abspath("../../scripts"))
print(os.sys.path)

import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential, load_model
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from nnom_utils import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_name = 'keras_mnist_trained_model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')

def build_model(x_shape):
    inputs = Input(shape=x_shape)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid')(inputs)
    #x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding="same")(x)

    """
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)

    
    x = Cropping2D(cropping=((3,2),(3,1)))(x)
    x = UpSampling2D(size=(2,2))(x)
    x = ZeroPadding2D(padding=((1, 2), (3, 4)))(x)
    """
    """
    x = Conv2D(24, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding="same")(x)

   
    x = Conv2D(24, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(24, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x2 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding="same")(x)
    x3 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)

    x = Concatenate(axis=-1)([x1, x2, x3])
    
    """
    x = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="valid")(x)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding="same")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="valid")(x)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding="same")(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(10)(x)
    predictions = Softmax()(x)

    return Model(inputs=inputs, outputs=predictions)


def train(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=50):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.summary()

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    # save best
    checkpoint = ModelCheckpoint(filepath=model_path,
            monitor='val_acc',
            verbose=0,
            save_best_only='True',
            mode='auto',
            period=1)
    callback_lists = [checkpoint]

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              shuffle=True, callbacks=callback_lists)

    del model
    K.clear_session()

    return history

def main():
    """
    # fixed the gpu error
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    """

    epochs = 1
    num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test_original, y_test_original) = mnist.load_data()

    x_test = x_test_original
    y_test = y_test_original
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # reshape to 4 d becaue we build for 4d?
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    print('x_train shape:', x_train.shape)

    # quantize the range to q7
    x_test = x_test/255
    x_train = x_train/255
    print("data range", x_test.min(), x_test.max())

    # generate binary dataset for NNoM validation, 0~1 -> 0~127, q7
    generate_test_bin(x_test*127, y_test, name='mnist_test_data.bin')

    # build model
    model = build_model(x_test.shape[1:])

    # train model
    history = train(model, x_train.copy(), y_train, x_test.copy(), y_test, epochs=epochs)

    # -------- generate weights.h (NNoM model) ----------
    # get the best model
    model_path = os.path.join(save_dir, model_name)
    model = load_model(model_path)

    # evaluate
    evaluate_model(model, x_test, y_test)

    # save weight
    generate_model(model,  x_test, format='hwc', name="weights.h")

    # --------- for test in CI ----------
    # build nnom
    os.system("scons")

    # do inference
    cmd = ".\mnist.exe" if 'win' in sys.platform else "./mnist"
    if(0 == os.system(cmd)):
        result = np.genfromtxt('result.csv', delimiter=',', skip_header=1)
        result = result[:,0] # the first column is the label, the second is the probability
        label = y_test_original
        acc = np.sum(result == label)/len(result)
        if (acc > 0.8):
            print("Top 1 Accuracy using NNoM  %.2f%%" %(acc *100))
            return 0
        else:
            raise Exception('test failed, accuracy is %.1f%% < 80%%' % (acc * 100.0))

if __name__ == "__main__":
    main()