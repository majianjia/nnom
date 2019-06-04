'''
    Copyright (c) 2018-2019
    Jianjia Ma, Wearable Bio-Robotics Group (WBR)
    majianjia@live.com

    SPDX-License-Identifier: LGPL-3.0

    Change Logs:
    Date           Author       Notes
    2019-02-05     Jianjia Ma   The first version


    This file provides:
    -> fake_quantisation layers which simulate the output quantisation on fixed-point NN models.
    -> weights/bias quantisation of Convolution and Dense Layer. "weight.h" file generations
    -> export "testing set" binary data file.
    -> print output ranges of each layers.

    Currently, this script does not support RNN (type) layers.
'''


import matplotlib.pyplot as plt
import os

from keras.models import Sequential, load_model
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from nnom_utils import *


def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

def one_hot(y_, n_classes=6):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def plot_raw(data):
    list = range(0, data.shape[0], int( data.shape[0]/10))
    for i in list:
        dat = data[int(i), :, :]
        print(dat.shape)
        plt.plot(dat)
        plt.title('raw')
        plt.show()

def normalize(data):
    data[:, :, 0:3] = data[:, :, 0:3] / max(abs(np.max(data[:, :, 0:3])), abs(np.min(data[:, :, 0:3])))
    data[:, :, 3:6] = data[:, :, 3:6] / max(abs(np.max(data[:, :, 3:6])), abs(np.min(data[:, :, 3:6])))
    data[:, :, 6:9] = data[:, :, 6:9] / max(abs(np.max(data[:, :, 6:9])), abs(np.min(data[:, :, 6:9])))
    return data



def train(x_train, y_train, x_test, y_test, batch_size= 64, epochs = 100):

    # shuffle
    permutation = np.random.permutation(y_train.shape[0])
    x_train = x_train[permutation, :, :]
    y_train = y_train[permutation]

    #plot_raw(x_test)

    print("x_train shape", x_train.shape)
    print("y_train shape", y_train.shape)

    inputs = Input(shape=x_train.shape[1:])
    x = Conv1D(32, kernel_size=(9), strides=(2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = ReLU()(x)
    x = MaxPool1D(2, strides=2)(x)

    # inception - 1
    x1 = Conv1D(32, kernel_size=(5), strides=(1), padding="same")(x)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.2)(x1)
    x1 = ReLU()(x1)
    x1 = MaxPool1D(2, strides=2)(x1)

    # inception - 2
    x2 = Conv1D(32, kernel_size=(3), strides=(1), padding="same")(x)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    x2 = ReLU()(x2)
    x2 = MaxPool1D(2, strides=2)(x2)

    # inception - 3
    x3 = MaxPool1D(2, strides=2)(x)
    x3 = Dropout(0.2)(x3)

    # concate all inception layers
    x = concatenate([ x1, x2,x3], axis=-1)

    # conclusion
    x = Conv1D(48, kernel_size=(3), strides=(1), padding="same")(x)
    x = ReLU()(x)
    x = MaxPool1D(2, strides=2)(x)
    x = Dropout(0.2)(x)

    # our netowrk is not that deep, so a hidden fully connected layer is introduce
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dropout(0.2)(x)
    x = ReLU()(x)
    x = Dense(6)(x)
    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(
                  loss='categorical_crossentropy',
                  optimizer='Adam',
                  #optimizer='Adadelta',
                  #optimizer='RMSprop',
                  #optimizer='SGD',
                  metrics=['accuracy'])

    model.summary()

    # save best
    checkpoint = ModelCheckpoint(filepath=MODEL_PATH,
            monitor='val_acc',
            verbose=0,
            save_best_only='True',
            mode='auto',
            period=1)
    callback_lists = [checkpoint]

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                        verbose=2, validation_data=(x_test, y_test), callbacks=callback_lists)

    # free the session to avoid nesting naming while we load the best model after.
    del model
    K.clear_session()
    return history



if __name__ == "__main__":
    ## cpu normally run fast in 1-D data
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if(not os.path.exists('data/UCI HAR Dataset')):
        raise Exception('Please download the dataset and unzip into "data/UCI HAR Dataset/"')

    epochs = 50

    # Those are separate normalised input features for the neural network
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

    # Output classes to learn how to classify
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    DATA_PATH = "data/"
    MODEL_PATH = 'best_model.h5'

    DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)

    TRAIN = "train/"
    TEST = "test/"

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]

    x_train = load_X(X_train_signals_paths)
    x_test = load_X(X_test_signals_paths)

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    y_train = one_hot(y_train, 6)
    y_test = one_hot(y_test, 6)

    # normolized each sensor, to range -1~1
    x_train = normalize(x_train)
    x_test  = normalize(x_test)

    # test, ranges
    print("train acc range", np.max(x_train[:, :, 0:3]), np.min(x_train[:, :, 0:3]))
    print("train acc2 range", np.max(x_train[:, :, 6:9]), np.min(x_train[:, :,6:9]))
    print("train gyro range", np.max(x_train[:, :, 3:6]), np.min(x_train[:, :, 3:6]))
    print("test acc range", np.max(x_test[:, :, 0:3]), np.min(x_test[:, :, 0:3]))
    print("test acc2 range", np.max(x_test[:, :, 6:9]), np.min(x_test[:, :,6:9]))
    print("test gyro range", np.max(x_test[:, :, 3:6]), np.min(x_test[:, :, 3:6]))

    # generate binary test data, convert range to [-128 127] for mcu
    x_test_bin = np.clip(x_test *128, -128, 127)
    x_train_bin = np.clip(x_train*128, -128, 127)
    generate_test_bin(x_test_bin, y_test, name='uci_test_data.bin')
    generate_test_bin(x_train_bin, y_train, name='uci_train_data.bin')

    # train model
    history = train(x_train,y_train, x_test, y_test, batch_size=128, epochs=epochs)

    # get best model
    model = load_model(MODEL_PATH)

    # evaluate
    evaluate_model(model, x_test, y_test)

    # save weight
    generate_model(model, x_test, 'weights.h')

    # plot
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.plot(range(0, epochs), acc, color='red', label='Training acc')
    plt.plot(range(0, epochs), val_acc, color='green', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
