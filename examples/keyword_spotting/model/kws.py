import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import os

import keras
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from nnom_utils import *

from mfcc import *

model_path = 'model.h5'

def mfcc_plot(x, label= None):
    mfcc_feat = np.swapaxes(x, 0, 1)
    ig, ax = plt.subplots()
    cax = ax.imshow(mfcc_feat, interpolation='nearest', origin='lower', aspect=1)#, cmap=cm.coolwarm)
    if label is not None:
        ax.set_title(label)
    else:
        ax.set_title('MFCC')
    plt.show()

def label_to_category(label, selected):
    category = []
    for word in label:
        if(word in selected):
            category.append(selected.index(word))
        else:
            category.append(len(selected)) # all others
    return np.array(category)

def train(x_train, y_train, x_test, y_test, type, batch_size=64, epochs=100):
    inputs = Input(shape=x_train.shape[1:])
    x = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 1), strides=(2, 1), padding="valid")(x)

    x = Conv2D(32 ,kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 1),strides=(2, 1), padding="valid")(x)

    x = Conv2D(64 ,kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #x = MaxPool2D((2, 1), strides=(2, 1), padding="valid")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(type)(x)

    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.summary()

    # save best
    checkpoint = ModelCheckpoint(filepath=model_path,
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

    del model
    K.clear_session()

    return history

def main():

    # fixed the gpu error
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    try:
        x_train = np.load('train_data.npy')
        y_train = np.load('train_label.npy')
        x_test = np.load('test_data.npy')
        y_test = np.load('test_label.npy')
        x_val = np.load('val_data.npy')
        y_val = np.load('val_label.npy')

    except:
        # test
        (x_train, y_train), (x_test, y_test), (x_val, y_val) = merge_mfcc_file()

        np.save('train_data.npy', x_train)
        np.save('train_label.npy', y_train)
        np.save('test_data.npy', x_test)
        np.save('test_label.npy', y_test)
        np.save('val_data.npy', x_val)
        np.save('val_label.npy', y_val)


    # label: the selected label will be recognised, while the others will be classified to "unknow". 
    #selected_lable = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    #selected_lable = ['marvin', 'sheila', 'yes', 'no', 'left', 'right', 'forward', 'backward', 'stop', 'go']

    selected_lable = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight','five', 'follow', 'forward',
                      'four','go','happy','house','learn','left','marvin','nine','no','off','on','one','right',
                      'seven','sheila','six','stop','three','tree','two','up','visual','yes','zero']

    # parameters
    epochs = 20
    batch_size = 64
    num_type = len(selected_lable)+1

    # only take 2~13 coefficient. 1 is destructive.
    x_train = x_train[:, :, 1:]
    x_test = x_test[:, :, 1:]
    x_val = x_val[:, :, 1:]

    # expand on channel axis because we only have one channel
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
    print('x_train shape:', x_train.shape, 'max', x_train.max(), 'min', x_train.min())

    # fake quantised
    # instead of using maximum value for quantised, we allows some saturation to save more details in small values.
    quantise_factor = pow(2, 4)
    print("quantised by", quantise_factor)

    x_train = (x_train / quantise_factor)
    x_test = (x_test / quantise_factor)
    x_val = (x_val / quantise_factor)

    # training data enforcement
    x_train = np.vstack((x_train, x_train*0.8))
    y_train = np.hstack((y_train, y_train))
    print(y_train.shape)

    # saturation to -1 to 1
    x_train = np.clip(x_train, -1, 1)
    x_test = np.clip(x_test, -1, 1)
    x_val = np.clip(x_val, -1, 1)

    # -1 to 1 quantised to 256 level (8bit)
    x_train = (x_train * 128).round()/128
    x_test = (x_test * 128).round()/128
    x_val = (x_val * 128).round()/128

    print('quantised', 'x_train shape:', x_train.shape, 'max', x_train.max(), 'min', x_train.min())
    print("dataset abs mean at", abs(x_test).mean()*128)

    # test, if you want to see a few random MFCC imagea. 
    if(0):
        which = 232
        while True:
            mfcc_plot(x_train[which].reshape((63, 12))*128, y_train[which])
            which += 352

    # word label to number label
    y_train = label_to_category(y_train, selected_lable)
    y_test = label_to_category(y_test, selected_lable)
    y_val = label_to_category(y_val, selected_lable)

    # number label to onehot
    y_train = keras.utils.to_categorical(y_train, num_type)
    y_test = keras.utils.to_categorical(y_test, num_type)
    y_val = keras.utils.to_categorical(y_val, num_type)

    # shuffle test data
    permutation = np.random.permutation(x_test.shape[0])
    x_test = x_test[permutation, :]
    y_test = y_test[permutation]
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation, :]
    y_train = y_train[permutation]

    # generate test data for MCU
    generate_test_bin(x_test * 127, y_test, 'test_data.bin')
    generate_test_bin(x_train * 127, y_train, 'train_data.bin')

    # do the job
    history = train(x_train, y_train, x_val, y_val, type=num_type, batch_size=batch_size, epochs=epochs)

    # reload the best model
    model = load_model(model_path)

    evaluate_model(model, x_test, y_test)

    generate_model(model, np.vstack((x_test, x_val)), name="weights.h")

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.plot(range(0, epochs), acc, color='red', label='Training acc')
    plt.plot(range(0, epochs), val_acc, color='green', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__=='__main__':
    main()