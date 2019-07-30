'''
    Copyright (c) 2018-2019
    Jianjia Ma, Wearable Bio-Robotics Group (WBR)
    majianjia@live.com
    SPDX-License-Identifier: Apache-2.0
    Change Logs:
    Date           Author       Notes
    2019-07-25     Jianjia Ma   The first version
'''

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

save_dir = 'keras_mnist_trained_model.h5'

def build_model(x_shape):
    inputs = Input(shape=x_shape)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding="same")(x)

    x = UpSampling2D()(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
    x = Cropping2D(((5,6),(7,8)))(x)


    x1 = Conv2D(12, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    x2 = Conv2D(12, kernel_size=(5, 5), strides=(1, 1), padding="same")(x)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    x = concatenate([x1, x2])
    x = Flatten()(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(10)(x)
    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def train(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=50):
    # save best
    checkpoint = ModelCheckpoint(filepath=save_dir,
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

    # fixed the gpu error
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)


    epochs = 10
    num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test_original, y_test_original) = cifar10.load_data()

    x_test = x_test_original
    y_test = y_test_original
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # reshape to 4 d becaue we build for 4d?
    #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    print('x_train shape:', x_train.shape)

    # quantize the range to q7
    x_test = x_test.astype('float32')/255
    x_train = x_train.astype('float32')/255
    print("data range", x_test.min(), x_test.max())

    # build model
    model = build_model(x_test.shape[1:])

    # train model
    history = train(model, x_train, y_train, x_test.copy(), y_test.copy(), epochs=epochs)

    # -------- generate weights.h (NNoM model) ----------
    # get the best model
    model = load_model(save_dir)

    # plotlayer output in keras
    L = model.layers
    test_img =  x_test[0].reshape(1, x_test.shape[1], x_test.shape[2], x_test.shape[3])
    for inx, layer in enumerate(L):  # layer loop
        import matplotlib.pyplot as plt
        if(model.input == layer.output or
            'dropout' in layer.name):
            continue
        layer_model = Model(inputs=model.input, outputs=layer.output)
        features = layer_model.predict(test_img).flatten()
        layer_name = layer.name.split('/')[0]
        plt.hist(features, bins=128)
        plt.savefig("tmp/" +'keras' +str(inx) + layer_name + ".png")
        plt.close()


    # generate binary dataset for NNoM validation, 0~1 -> 0~127, q7
    (x_test[0]*127).tofile("tmp/input.raw")

    # generate NNoM model, x_test is the calibration dataset used in quantisation process
    generate_model(model,  x_test[:50], format='hwc', name="weights.h")

    # --------- for test in CI ----------
    # build NNoM
    os.system("scons")

    # do inference using NNoM
    cmd = ".\mnist.exe" if 'win' in sys.platform else "./mnist"
    if(0 == os.system(cmd)):
        import matplotlib.pyplot as plt
        import glob
        for inx, filename in enumerate(glob.glob('tmp/*.raw')):
            result = np.fromfile(filename, dtype="int8")
            plt.hist(result, bins=128)
            plt.savefig(filename + ".png")
            plt.close()



if __name__ == "__main__":
    main()