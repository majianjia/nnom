'''
    Copyright (c) 2018-2019
    Jianjia Ma, Wearable Bio-Robotics Group (WBR)
    majianjia@live.com

    SPDX-License-Identifier: LGPL-3.0

    Change Logs:
    Date           Author       Notes
    2019-02-12     Jianjia Ma   The first version

'''


import matplotlib.pyplot as plt
import os

from keras.models import Sequential, load_model
from keras.models import Model
from keras.datasets import mnist
from keras.datasets import cifar10 #test
from keras.layers import *
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from nnom_utils import *


model_name = 'mnist_model.h5'
save_dir = model_name #os.path.join(os.getcwd(), model_name)

def dense_block(x, k):

    x1 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x)
    x1 = fake_clip()(x1)
    x1 = ReLU()(x1)

    x2 = concatenate([x, x1],axis=-1)
    x2 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x2)
    x2 = fake_clip()(x2)
    x2 = ReLU()(x2)

    x3 = concatenate([x, x1, x2],axis=-1)
    x3 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x3)
    x3 = fake_clip()(x3)
    x3 = ReLU()(x3)

    x4 = concatenate([x, x1, x2, x3],axis=-1)
    x4 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x4)
    x4 = fake_clip()(x4)
    x4 = ReLU()(x4)

    return concatenate([x, x1, x2, x3, x4],axis=-1)

def train(x_train, y_train, x_test, y_test, batch_size= 64, epochs = 100):

    inputs = Input(shape=x_train.shape[1:])
    x = Conv2D(16, kernel_size=(7, 7), strides=(1, 1), padding='same')(inputs)
    x = fake_clip()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2),strides=(2, 2), padding="same")(x)

    # dense block
    x = dense_block(x, k=12)

    # bottleneck -1
    x = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = fake_clip()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding="same")(x)

    # dense block -2
    x = dense_block(x, k=24)

    x = Conv2D(10, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = fake_clip()(x)
    x = ReLU()(x)

    # global avg.
    x = GlobalAvgPool2D()(x)
    x = fake_clip()(x)

    """
    # output
    #x = Flatten()(x)
    x = Dense(128)(x)
    x = fake_clip()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    x = Dense(10)(x)
    x = fake_clip()(x)
    """
    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    # save best
    checkpoint = ModelCheckpoint(filepath=save_dir,
            monitor='val_acc',
            verbose=0,
            save_best_only='True',
            mode='auto',
            period=1)
    callback_lists = [checkpoint]

    history =  model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              shuffle=True, callbacks=callback_lists)
			  
    # free the session to avoid nesting naming while we load the best model after.
    del model
    K.clear_session()
    return history


if __name__ == "__main__":

    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    epochs = 5
    num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # reshape to 4 d becaue we build for 4d?
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    print('x_train shape:', x_train.shape)

    # quantize the range to q7 without bias
    x_test = np.clip(np.floor((x_test)/8), -128, 127)
    x_train = np.clip(np.floor((x_train)/8), -128, 127)

    print("data range", x_test.min(), x_test.max())

    # generate binary
    generate_test_bin(x_test, y_test, name='mnist_test_data.bin')

    # train model
    history = train(x_train,y_train, x_test, y_test, batch_size=128, epochs=epochs)

    # get best model
    model = load_model(save_dir)

    # evaluate
    evaluate_model(model, x_test, y_test)

    # save weight
    generate_weights(model, name='weights.h')

    # test, show the output ranges
    layers_output_ranges(model, x_train)


    # plot
    if(1):
        acc = history.history['acc']
        val_acc = history.history['val_acc']

        plt.plot(range(0, epochs), acc, color='red', label='Training acc')
        plt.plot(range(0, epochs), val_acc, color='green', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()





















