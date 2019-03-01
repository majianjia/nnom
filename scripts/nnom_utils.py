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


import tensorflow as tf
from keras.layers import Lambda
from keras.models import Model
from keras import backend as K
from sklearn import metrics
from fully_connected_opt_weight_generation import *
import time


""" 
    This lambda layer take the output variables (vectors) from last layer, 
    clip range to clip_range
    quantize to the bits.
"""
def quant_layer(x, clip_range, bits):
    import tensorflow as tf
    return tf.fake_quant_with_min_max_vars(x, min=clip_range[0], max=clip_range[1], num_bits=bits)


def quant_shape(input_shape):
    return input_shape


def fake_clip(frac_bit=0, bit=8):
    '''
    :param frac_bit:  fractional bit number. Q3.4 = shift 4
    :param bit: width
    :return:
    '''
    max = 2**(bit - frac_bit) / 2 - (1/(2**frac_bit))
    min = -2**(bit - frac_bit) / 2
    return Lambda(quant_layer, output_shape=quant_shape, arguments={'clip_range': [min, max], 'bits': bit})

def fake_clip_min_max(min=0, max =1,  bit=8):
    return Lambda(quant_layer, output_shape=quant_shape, arguments={'clip_range': [min, max], 'bits': bit})

""" 
this is the generate the test set data to a bin file
bin file can be used to validate the implementation in MCU

"""

def generate_test_bin(x, y, name='test_data_with_label.bin'):
    '''
    this method generate the
    :param x:  input x data size
    :param y:  input label (one hot label)
    :return:
    '''
    # get label
    test_label = np.argwhere(y == 1).astype(dtype="byte")  # test data
    test_label = test_label[:, 1]

    # get test data
    dat = x.astype(dtype="byte")  # test data
    block_size = x.shape[1] * x.shape[2] * x.shape[3] # size of one sample for exampe, mnist = 28*28*1
    dat = np.reshape(dat, (dat.shape[0] * block_size))

    label_batch = 128       # the Y-modem example uses 128 batch

    with open(name, 'wb') as f:
        start = 0
        while start <= (test_label.size - label_batch):
            test_label[start: start + label_batch].tofile(f)
            dat[block_size * start: block_size * (start + label_batch)].tofile(f)
            start += label_batch

        # the rest data
        if (start < test_label.size):
            rest_len = test_label.size - start
            new_labls = test_label[start:]
            new_labls = np.pad(new_labls, (0, label_batch - rest_len), mode='constant')
            new_labls.tofile(f)
            dat[block_size * start:].tofile(f)

    print("binary test file generated:", name)
    print("test data length:", test_label.size)
    return


def generate_weights(model, name='weights.h'):
    # Quantize weights to 8-bits using (min,max) and write to file
    f = open('weights.h', 'wb')
    f.close()

    for layer in model.layers:
        if (not layer.weights):
            continue

        weight_dec_shift = 0
        for var in layer.weights:
            var_name = str(var.name)
            if("kernel" in var_name):
                var_values = layer.get_weights()[0] # weight
                print("weight:", var_name)
                #print(var_values)
            else:
                var_values = layer.get_weights()[1] # bias
                print("bias: ",var_name)
            print("original shape: ", var_values.shape)

            min_value = np.min(var_values)
            max_value = np.max(var_values)

            int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
            dec_bits = 7 - int_bits
            print("original dec bit", dec_bits)

            # check if bias shift > weight shift, then reduce bias shift to weight shift
            if ("kernel" in var_name):
                weight_dec_shift = dec_bits
            else:
                if(dec_bits > weight_dec_shift):
                    dec_bits = weight_dec_shift
            print("new dec bit", dec_bits)

            # convert to [-128,128) or int8
            var_values = np.round(var_values * 2 ** dec_bits)
            var_name = var_name.replace('/', '_')
            var_name = var_name.replace(':', '_')
            with open(name, 'a') as f:
                f.write('#define ' + var_name.upper() + ' {')

            if (len(var_values.shape) == 3):  # 1D convolution layer weights
                transposed_wts = np.transpose(var_values, (2, 0, 1))
                #transposed_wts = var_values
            elif (len(var_values.shape) > 2):  # 2D convolution layer weights
                transposed_wts = np.transpose(var_values, (3, 0, 1, 2))
            else:  # fully connected layer weights or biases of any layer
                # test, use opt weight reorder
                if "dense" in var_name and "kernel" in var_name:
                    transposed_wts = np.transpose(var_values)
                    transposed_wts = convert_to_x4_q7_weights(np.reshape(transposed_wts ,(transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
                else:
                    transposed_wts = np.transpose(var_values)

            print("reshape to:",transposed_wts.shape)

            with open(name, 'a') as f:
                transposed_wts.tofile(f, sep=", ", format="%d")
                f.write('}\n\n')
                if ("bias" in var_name):
                    f.write('#define ' + var_name.upper() + '_SHIFT ' + '(' + str(dec_bits) + ')\n\n\n')
                if ("kernel" in var_name):
                    f.write('#define ' + var_name.upper() + '_SHIFT ' + '(' + str(dec_bits) + ')\n\n')

            with K.tf.Session() as session:
                # convert back original range but quantized to 8-bits or 256 levels
                var_values = var_values / (2 ** dec_bits)
                var_values = session.run(K.tf.assign(var, var_values))
                print(var_name + ' number of wts/bias: ' + str(var_values.shape) + \
                  ' dec bits: ' + str(dec_bits) + \
                  ' max: (' + str(np.max(var_values)) + ',' + str(max_value) + ')' + \
                  ' min: (' + str(np.min(var_values)) + ',' + str(min_value) + ')')


def generate_weights_outputshift(model, name='weights.h'):
    # Quantize weights to 8-bits using (min,max) and write to file
    f = open('weights.h', 'wb')
    f.close()

    dict = {}
    index = 0
    for layer in model.layers:
        if (not layer.weights):
            continue
        # use a dictionary to store configs
        dict_name = layer.name
        dict[dict_name] = {}

        weight_dec_shift = 0
        for var in layer.weights:
            var_name = str(var.name)
            if("kernel" in var_name):
                var_values = layer.get_weights()[0] # weight
                print("weight:", var_name)
            else:
                var_values = layer.get_weights()[1] # bias
                print("bias: ",var_name)

            min_value = np.min(var_values)
            max_value = np.max(var_values)

            int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
            dec_bits = 7 - int_bits
            print("original dec bit", dec_bits)

            # check if bias shift > weight shift, then reduce bias shift to weight shift
            if ("kernel" in var_name):
                weight_dec_shift = dec_bits
            else:
                if(dec_bits > weight_dec_shift):
                    dec_bits = weight_dec_shift
            print("new dec bit", dec_bits)

            # convert to [-128,128) or int8
            var_values = np.round(var_values * 2 ** dec_bits)
            var_name = var_name.replace('/', '_')
            var_name = var_name.replace(':', '_')
            with open(name, 'a') as f:
                f.write('#define ' + var_name.upper() + ' {')

            if (len(var_values.shape) == 3):  # 1D convolution layer weights
                transposed_wts = np.transpose(var_values, (2, 0, 1))

            elif (len(var_values.shape) > 2):  # 2D convolution layer weights
                transposed_wts = np.transpose(var_values, (3, 0, 1, 2))

            else:  # fully connected layer weights or biases of any layer
                # test, use opt weight reorder
                if "dense" in var_name and "kernel" in var_name:
                    transposed_wts = np.transpose(var_values)
                    transposed_wts = convert_to_x4_q7_weights(np.reshape(transposed_wts ,(transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
                else:
                    transposed_wts = np.transpose(var_values)

            # save to dict
            dict[dict_name][var_name] = transposed_wts
            dict[dict_name][var_name+"_shift"] = dec_bits

    # all weight is quantized and saved to dictionary
    print(dict)

def layers_output_ranges(model, x_test):
    # test, show the output ranges
    shift_list = np.array([])
    for layer in model.layers:
        if("input" in layer.name
                or "dropout" in layer.name
                or "softmax" in layer.name
                or "lambda" in layer.name
                or "concat" in layer.name
                or "re_lu" in layer.name):
            continue
        layer_model = Model(inputs=model.input, outputs=layer.output)
        features = layer_model.predict(x_test)
        max_val = features.max()
        min_val = features.min()
        print( layer.name)
        print("         max value:", max_val)
        print("         min value:", min_val)
        int_bits = int(np.ceil(np.log2(max(abs(max_val), abs(min_val)))))
        dec_bits = 7 - int_bits
        print("         dec bit", dec_bits)
        # record the shift
        shift_list = np.append(shift_list, dec_bits)

    print("shift list", shift_list)



def evaluate_model(model, x_test, y_test, running_time=False, to_file='evaluation.txt'):
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', scores[0])
    print('Top 1:', scores[1])

    predictions = model.predict(x_test)
    output = tf.keras.metrics.top_k_categorical_accuracy(y_test, predictions, k=2)
    with tf.Session() as sess:
        result = sess.run(output)
    print("Top 2:",result)

    run_time = 0
    if running_time:
        # try to calculate the time
        T = time.time()
        for i in range(10):
            model.predict(x_test)
        T = time.time() - T
        run_time = round((T / 10 / x_test.shape[0] * 1000 * 1000), 2)
        print("Runing time:",run_time , "us" )

    predictions = model.predict(x_test)
    matrix = metrics.confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    print(matrix)
    #
    with open(to_file, 'w') as f:
        f.write('Test loss:'+ str(scores[0]) + "\n")
        f.write('Top 1:'+ str(scores[1])+ "\n")
        f.write("Top 2:"+ str(result)+ "\n")
        f.write("Runing time: "+ str(run_time) + "us" + "\n")
        f.write(str(matrix))


    # try to check the weight and bias dec ranges
    for layer in model.layers:
        if (not layer.weights):
            continue

        for var in layer.weights:
            var_name = str(var.name)
            if ("kernel" in var_name):
                var_values = layer.get_weights()[0]  # weight
            else:
                var_values = layer.get_weights()[1]  # bias
            min_value = np.min(var_values)
            max_value = np.max(var_values)
            intt = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
            dec = 7 - intt
            print(var_name, "Dec num:", dec)



class nnom:
    def __init__(self):
        self.shift_dict = {}
        return

    def __enter__(self):
        self.shift_dict = {}
        return

    def __exit__(self, type, value, traceback):
        print("output shift",dict)
        np.save('output_shifts.npy', self.shift_dict)


    # to record the shift range for interested layers (outputs)
    shift_dict = {}

    def fake_clip(self, x, frac_bit=0, bit=8):
        name = str(x.name)
        self.shift_dict[name] = frac_bit
        return fake_clip(frac_bit, bit)

    def fake_clip_min_max(self, output_layer_name, min=0, max=1,  bit=8):
        self.shift_dict[output_layer_name] = bit - 1 - np.int(np.ceil(np.log2(np.max(abs(min), abs(max)))))
        return fake_clip_min_max(min=min, max=max, bit=bit)

    def save_shift(self, file='output_shifts.npy'):
        print("output shift", self.shift_dict)
        np.save(file, self.shift_dict)


# for test only
if __name__ == "__main__":
    import os
    from keras.models import  load_model

    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    # get best model
    #MODEL_PATH = 'best_model.h5'
    #model = load_model(MODEL_PATH)
    # save weight
    #generate_weights_test(model)





































