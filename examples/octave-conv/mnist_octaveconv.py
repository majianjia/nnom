'''
    Copyright (c) 2018-2020
    Jianjia Ma
    majianjia@live.com
    SPDX-License-Identifier: Apache-2.0
    Change Logs:
    Date           Author       Notes
    2019-02-12     Jianjia Ma   The first version
'''


import matplotlib.pyplot as plt
import os
nnscript = os.path.abspath('../../scripts')
os.sys.path.append(nnscript)

from tensorflow.keras import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model, save_model
import tensorflow as tf
import numpy as np
from nnom import *

model_name = 'model.h5'

def image_to_cfile(data, label, num_of_image, file='image.h'):
    # test
    with open(file, 'w') as f:
        for i in range(num_of_image):
            selected = np.random.randint(0, 1000) # select 10 out of 1000.
            f.write('#define IMG%d {'% (i))
            np.round(data[selected]).flatten().tofile(f, sep=", ", format="%d") # convert 0~1 to 0~127
            f.write('} \n')
            f.write('#define IMG%d_LABLE'% (i))
            f.write(' %d \n \n' % label[selected])
        f.write('#define TOTAL_IMAGE %d \n \n'%(num_of_image))

        f.write('static const int8_t img[%d][%d] = {' % (num_of_image, data[0].flatten().shape[0]))
        f.write('IMG0')
        for i in range(num_of_image -1):
            f.write(',IMG%d'%(i+1))
        f.write('};\n\n')

        f.write('static const int8_t label[%d] = {' % (num_of_image))
        f.write('IMG0_LABLE')
        for i in range(num_of_image -1):
            f.write(',IMG%d_LABLE'%(i+1))
        f.write('};\n\n')


def octave_conv2d(xh, xl, ch=12):
    # one octave convolution is consist of the 2 equations
    # YH=f(XH;WH→H)+upsample(f(XL;WL→H),2)
    # YL=f(XL;WL→L)+f(pool(XH,2);WH→L))

    # f(XL;WL→L)
    xhh = Conv2D(ch, kernel_size=(3, 3), strides=(1, 1), padding='same')(xh)

    # f(XH;WH→H)
    xll = Conv2D(ch, kernel_size=(3, 3), strides=(1, 1), padding='same')(xl)

    # upsample(f(XL;WL→H),2)
    xlh = Conv2D(ch, kernel_size=(3, 3), strides=(1, 1), padding='same')(xl)
    xlh = UpSampling2D(size=(2, 2))(xlh)

    # f(pool(XH,2);WH→L))
    xhl = Conv2D(ch, kernel_size=(3, 3), strides=(1, 1), padding='same')(xh)
    xhl = MaxPool2D(pool_size=(2, 2), padding='same')(xhl)
    #xhl = AvgPool2D(pool_size=(2, 2), padding='same')(xhl)

    # yh = xhh + xlh
    # yl = xll + xhl
    yh = add([xhh, xlh])
    yl = add([xll, xhl])

    return yh, yl

def train(x_train, y_train, x_test, y_test, batch_size= 64, epochs = 100):

    inputs = Input(shape=x_train.shape[1:])
    x = Conv2D(12, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    xh = ReLU()(x)
    xl = MaxPool2D((2,2),strides=(2,2), padding="same")(x)

    # octa 1
    xh, xl = octave_conv2d(xh, xl, 12)

    # max pool
    xh = MaxPool2D()(xh)
    xl = MaxPool2D()(xl)

    # octa 2
    xh, xl = octave_conv2d(xh, xl, 12)

    # reduce xh dimention to fit xl
    xh = MaxPool2D()(xh)

    x = concatenate([xh, xl], axis=-1)

    # reduce size
    x = Conv2D(12, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)

    x = Flatten()(x)
    x = Dense(96)(x)
    x = Dropout(0.2)(x)

    x = ReLU()(x)
    x = Dense(10)(x)
    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    history =  model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              shuffle=True)

    # free the session to avoid nesting naming while we load the best model after.
    save_model(model, model_name)
    del model
    tf.keras.backend.clear_session()
    return history


if __name__ == "__main__":
    epochs = 5
    num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train_num), (x_test, y_test_num) = mnist.load_data()

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train_num, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test_num, num_classes)

    # reshape to 4 d becaue we build for 4d?
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    print('x_train shape:', x_train.shape)

    # quantize the range to 0~255 -> 0~1
    x_test = x_test/255
    x_train = x_train/255
    print("data range", x_test.min(), x_test.max())

    # select a few image and write them to image.h
    image_to_cfile(x_test*127, y_test_num, 10, file='image.h')

    # train model, save the best accuracy model
    history = train(x_train, y_train, x_test, y_test, batch_size=64, epochs=epochs)

    # reload best model
    model = load_model(model_name)

    # evaluate
    evaluate_model(model, x_test, y_test)

    # save weight
    generate_model(model, np.vstack((x_train, x_test)), name="weights.h")

    # plot
    acc = history.history['accuracy']           # please change to "acc" in tf1
    val_acc = history.history['val_accuracy']   # please change to "val_acc" in tf1

    plt.plot(range(0, epochs), acc, color='red', label='Training acc')
    plt.plot(range(0, epochs), val_acc, color='green', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()





















