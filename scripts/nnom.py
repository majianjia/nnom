'''
    Copyright (c) 2018-2020
    Jianjia Ma
    majianjia@live.com

    SPDX-License-Identifier: Apache-2.0

    Change Logs:
    Date           Author       Notes
    2019-02-05     Jianjia Ma   The first version
'''

import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import *
from tensorflow.keras.layers import *
from fully_connected_opt_weight_generation import *
from gen_config import *
import scipy.stats
import time
import warnings

model_major_version = 0
model_sub_version = 4
model_reversion = 2

#define NNOM_MAJORVERSION     0L              /**< major version number */
#define NNOM_SUBVERSION       4L              /**< minor version number */
#define NNOM_REVISION         2L              /**< revise version number */
#define NNOM_VERSION          (NNOM_MAJORVERSION * 10000) + (NNOM_SUBVERSION * 100) + NNOM_REVISION)

def fuse_bn_to_conv(layer):
    # try to fuse BN layer to convolutional
    if ('conv' in layer.name) and \
            ('batch_normalization' in layer.outbound_nodes[0].outbound_layer.name):
        print("fusing batch normalization to", layer.name)
        bn_layer = layer._outbound_nodes[0].outbound_layer
        c_w = layer.get_weights()[0]
        c_b = layer.get_weights()[1]
        print('original weight max', c_w.max(), 'min', c_w.min())
        print('original bias max', c_b.max(), 'min', c_b.min())
        bn_gamma = bn_layer.get_weights()[0]
        bn_beta = bn_layer.get_weights()[1]
        bn_mean = bn_layer.get_weights()[2]
        bn_variance = bn_layer.get_weights()[3]
        epsilon = 1e-3  # default epsilon for tf.slim.batch_norm
        if ('conv2d' in layer.name):
            if "depthwise" in layer.name:  # depthwise batchnorm params are ordered differently
                for l in range(c_w.shape[3]):
                    for k in range(c_w.shape[2]):
                        for j in range(c_w.shape[1]):
                            for i in range(c_w.shape[0]):
                                c_w[i][j][k][l] *= bn_gamma[k*c_w.shape[3]+l] / np.sqrt(bn_variance[k*c_w.shape[3]+l] + epsilon)
                depth_dim = c_w.shape[2] * c_w.shape[3]  # test needed
            # normal conv
            else:
                for l in range(c_w.shape[3]):
                    for k in range(c_w.shape[2]):
                        for j in range(c_w.shape[1]):
                            for i in range(c_w.shape[0]):
                                c_w[i][j][k][l] *= bn_gamma[l] / np.sqrt(bn_variance[l] + epsilon)
                depth_dim = c_w.shape[3]
            for l in range(depth_dim):
                c_b[l] = (bn_gamma[l] * (c_b[l] - bn_mean[l]) / np.sqrt(bn_variance[l] + epsilon)) + bn_beta[l]
        # conv1d
        else:
            epsilon = 1e-3  # default epsilon for tf.slim.batch_norm
            for k in range(c_w.shape[2]):
                for j in range(c_w.shape[1]):
                    for i in range(c_w.shape[0]):
                        if "depthwise" in layer.name:  # depthwise batchnorm params are ordered differently
                            c_w[i][j][k] *= bn_gamma[j] / np.sqrt(bn_variance[j] + epsilon)
                        else:
                            c_w[i][j][k] *= bn_gamma[k] / np.sqrt(bn_variance[k] + epsilon)

            if "depthwise" in layer.name:
                depth_dim = c_w.shape[1]*c_w.shape[2] # need to be tested
            else:
                depth_dim = c_w.shape[2]
            for l in range(depth_dim):
                c_b[l] = (bn_gamma[l] * (c_b[l] - bn_mean[l]) / np.sqrt(bn_variance[l] + epsilon)) + bn_beta[l]

        print('fused weight max', c_w.max(), 'min', c_w.min())
        print('fused bias max', c_b.max(), 'min', c_b.min())
        # write the weights back to the layer
        # after that, the model will be destroyed.. need a better way to pass the new weight
        layer.set_weights([c_w, c_b])

def generate_test_bin(x, y, name='test_data_with_label.bin'):
    '''
    this method generate the
    :param x:  input x data size
    :param y:  input label (one hot label)
    :return:
    '''
    # quantize input x
    dec_bits = find_dec_bits_max_min(x, bit_width=8)
    x = np.round(x*2**dec_bits).astype(np.int8)
    # get label
    if(len(y.shape) >1):
        test_label = np.argwhere(y == 1).astype(np.int8)  # test data
        test_label = test_label[:, 1]
    else:
        test_label = y

    # get data
    dat = x.astype(dtype="byte")  # test data
    batch_size = dat.shape[0]     # total pices of data
    dat = dat.flatten()           # flatten to get the total size.
    block_size = int(dat.size / batch_size) # this must be integer but... just to confirm

    # write (label x 128) (data_block x 128)
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

def is_shift_layer(layer):
    ''' layer which can change the output encoding'''
    #FIXME: add more which will change the output shift
    if('input' in layer.name or
       'conv2d' in layer.name or
       'conv1d' in layer.name or
       'dense' in layer.name or
       'softmax' in layer.name or
        'sigmoid' in layer.name or
        'tanh' in layer.name or
        ('add' in layer.name and 'zero' not in layer.name) or # the name, zero_padding contains 'add'
        'subtract' in layer.name or
        'multiply' in layer.name or
       ('activation' in layer.name and layer.get_config()['activation'] == 'softmax')or
        ('activation' in layer.name and layer.get_config()['activation'] == 'hard_sigmoid') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'tanh') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'hard_tanh') or
        is_rnn_layer(layer)
    ):
        return True
    return False

def is_shift_fixed(layer):
    ''' layer which shift to a fixed value'''
    #FIXME: add more which will change the output shift
    if('softmax' in layer.name or
        'sigmoid' in layer.name or
        'tanh' in layer.name or
        ('activation' in layer.name and layer.get_config()['activation'] == 'softmax') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'sigmoid') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'hard_sigmoid') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'tanh') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'hard_tanh') or
        is_rnn_layer(layer)
    ):
        return True
    return  False

def is_lstm_layer(layer):
    if type(layer) is LSTM or 'lstm' in layer.name:
        return True
    if(type(layer) is RNN or 'rnn' in layer.name):
        if(type(layer.cell) is LSTMCell or 'lstm' in layer.cell.name):
            return True
    return False

def is_gru_layer(layer):
    if type(layer) is GRU or 'gru' in layer.name:
        return True
    if(type(layer) is RNN or 'rnn' in layer.name):
        if(type(layer.cell) is GRUCell or 'gru' in layer.cell.name):
            return True
    return False

def is_rnn_layer(layer):
    if( 'rnn' in layer.name or
        is_lstm_layer(layer) or
        is_gru_layer(layer)
    ):
        return True
    return  False

def find_offset(data):
    """
    Offset of the original data before quantisation
    :param data:
    :return: offset of the data block
    """
    return np.average(data)


def find_dec_bits_max_min(data, bit_width=8, maximum_bit=16):
    """
    A ragular non-saturated shift-based quantisation mathod. Using max/min values
    :param data:
    :param bit_width:
    :param maximum_bit: maximum decimal bit. Incase sometime bias is too small lead to very large size dec bit
    :return:
    """
    max_val = abs(data.max()) - abs(data.max()/pow(2, bit_width)) # allow very small saturation.
    min_val = abs(data.min()) - abs(data.max()/pow(2, bit_width))
    int_bits = int(np.ceil(np.log2(max(max_val, min_val))))
    dec_bits = (bit_width-1) - int_bits
    return min(dec_bits, maximum_bit)

def find_dec_bits_max_min_axis(data, axis=-1,bit_width=8, maximum_bit=16):
    """
    A ragular non-saturated shift-based quantisation mathod. Using max/min values
    :param data:
    :param axis:
    :param bit_width:
    :return:
    """
    dec_bits = []
    # if(len(data.shape) < np.abs(axis)): # for depthwise with axis = -2 while len(shape) =1
    #     size = data.shape[0]
    #     axis = 0 #
    # else:
    #     size = data.shape[axis]
    for i in np.arange(0, data.shape[axis]):
        d = np.take(data, indices=i, axis=axis)
        max_val = d.max()
        min_val = d.min()
        int_bit = int(np.ceil(np.log2(max(abs(max_val), abs(min_val)))))
        dec_bit = (bit_width-1) - int_bit
        dec_bits.append(min(dec_bit, maximum_bit))
    return dec_bits

def find_dec_bits_kld(data, bit_width=8, scan_times=4, maximum_bit=16):
    """
    # saturation shift, using KLD method (Kullback-Leibler divergence)
    # Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    :param data: The data for looking for quantisation
    :param bit_width: the bitwidth of the data
    :param scan_times: the times to try the best kld (normally the second is the best.)
    :return: dec bit width for this data
    """
    # do a regular non-saturated quantisation
    max_val = data.max()
    min_val = data.min()
    abs_max = max(abs(max_val), abs(min_val))
    int_bits = int(np.ceil(np.log2(max(abs(max_val), abs(min_val)))))
    dec_bits = (bit_width-1) - int_bits

    # now looking for the best quantisation using KLD method
    small_var = 1e-5
    bins = np.arange(-abs_max, abs_max, abs_max / 2048 * 2)
    q_bins = np.arange(-abs_max, abs_max, abs_max / 256 * 2)
    flat_hist = np.histogram(data.flatten(), bins=bins)[0]
    kl_loss = []
    kl_shifts = []
    for shift in range(scan_times):
        t = 2 ** (dec_bits  + shift)  # 2-based threshold
        act = np.round(data.flatten() * t)
        act = act / t
        act = np.clip(act, -128 / t, 127 / t)
        act = np.histogram(act, bins=q_bins)[0]
        act_hist = np.zeros(2047)
        chunk = int(2048 / 256)
        for i in range(int(255)):
            none_zero = np.count_nonzero(flat_hist[i * chunk:(i + 1) * chunk])
            if none_zero == 0:
                continue
            for j in range(chunk):
                act_hist[i * chunk + j] = act[i] / none_zero if flat_hist[i * chunk + j] != 0 else 0
        flat_hist[flat_hist == 0] = small_var
        act_hist[act_hist == 0] = small_var
        kl = scipy.stats.entropy(flat_hist, act_hist)
        kl_loss.append(kl)
        kl_shifts.append(dec_bits + shift)

    # now get the least loss from the scaned kld shift
    dec_bits = kl_shifts[np.argmin(kl_loss)]  # set the dec_bit to the KLD results
    return min(dec_bits, maximum_bit)

# convert to [-128,128) or int8
def quantize_data(data, dec_bits, axis=-1, per_axis=False, bitwith=8):
    if (per_axis):
        out = []
        for i in np.arange(0, data.shape[axis]):
            d = np.take(data, indices=i, axis=axis)
            d = np.round(d * 2 ** dec_bits[i])
            d = np.clip(d, -2**(bitwith-1), 2**(bitwith-1)-1)
            d = np.expand_dims(d, axis=axis)
            out.append(d)
        out = np.concatenate(out, axis=axis)
        return out
    else:
        return np.clip(np.round(data * 2 ** dec_bits), -2**(bitwith-1), 2**(bitwith-1) -1)

def quantize_rnn_intermediate_output(layer, features):
    def nnom_sigmoid(data):
        return 1 / (1 + np.exp(-data))
    def nnom_tanh(data):
        return np.tanh(data)
    def split_array(d, num):
        l = len(d)
        if(num==4):
            return d[:int(l/4)], d[int(l/4): int(l/2)], d[int(l/2):-int(l/4)], d[-int(l/4):]
        elif(num==3):
            return d[:int(l/3)], d[int(l/3): -int(l/3)], d[-int(l/3):]
    lcfg = layer.get_config()
    if(lcfg['go_backwards']):
        features = features[:,::-1,:] # reverse timestamp

    if(type(layer.cell) is SimpleRNNCell):
        cfg = layer.cell.get_config()
        state = np.zeros(cfg['units'])
        kernel = layer.get_weights()[0]
        recurrent_kernel = layer.get_weights()[1]
        bias = layer.get_weights()[2]
        # replicate keras's implementation
        def simple_cell_step(inputs, state, kernel, recurrent_kernel, bias, activation):
            h = np.dot(inputs, kernel)
            h = np.add(h, bias)
            h2 = np.dot(state, recurrent_kernel)
            output = h + h2
            output = activation(output)
            return output, h, h2
        output_arrary = []
        h_array = []
        h2_array = []
        activation = nnom_tanh if cfg['activation'] is 'tanh' else nnom_sigmoid
        state = np.zeros(cfg['units'])
        for feature in features:
            if(not layer.stateful):
                state = np.zeros(cfg['units'])
            for fe in feature:
                output, h, h2 = simple_cell_step(fe, state, kernel, recurrent_kernel, bias, activation)
                state = output
                output_arrary.append(output)
                h_array.append(h)
                h2_array.append(h2)
        output_arrary = np.array(output_arrary)
        h_array = np.array(h_array)
        h2_array = np.array(h2_array)
        # qout = find_dec_bits_kld(output_arrary)
        # qh = find_dec_bits_kld(h_array)
        # qh2 = find_dec_bits_kld(h2_array)
        qout = find_dec_bits_max_min(output_arrary)
        qh = find_dec_bits_max_min(h_array)
        qh2 = find_dec_bits_max_min(h2_array)
        return [qout, qh, qh2]

    elif (type(layer.cell) is LSTMCell or 'lstm' in layer.cell.name):
        cfg = layer.cell.get_config()
        state = np.zeros(cfg['units']*2)
        kernel = layer.get_weights()[0]
        recurrent_kernel = layer.get_weights()[1]
        bias = layer.get_weights()[2]
        def lstm_cell_step(cell_inputs, cell_states, kernel, recurrent_kernel, bias):
            h_tm1 = cell_states[0]  # previous memory state
            c_tm1 = cell_states[1]  # previous carry state
            z1 = np.dot(cell_inputs, kernel)
            z1 = np.add(z1, bias)
            z2 = np.dot(h_tm1, recurrent_kernel)
            z = z1+z2               # -----> q_z
            z0, z1, z2, z3 = split_array(z, 4)
            i = nnom_sigmoid(z0) # q0.7
            f = nnom_sigmoid(z1) # q0.7
            c1 = f*c_tm1
            c2 = i*nnom_tanh(z2) # q0.7
            c = c1 + c2          # -----> q_c
            o = nnom_sigmoid(z3) # q0.7
            tc = nnom_tanh(c)
            h = o * tc # q0.7
            return h, [h, c], z ,z0, z1, z2, z3
        h_array = []
        c_array = []
        z_array = []
        z0_array = []
        z1_array = []
        z2_array = []
        z3_array = []
        state = [np.zeros(cfg['units']), np.zeros(cfg['units'])]
        for feature in features:
            if(not layer.stateful):
                state = [np.zeros(cfg['units']), np.zeros(cfg['units']) ]
            for fe in feature:
                output, state, z, z0, z1, z2, z3 = lstm_cell_step(fe, state, kernel, recurrent_kernel, bias)
                h_array.append(output)
                c_array.append(state[1])
                z_array.append(z)
                z0_array.append(z0)
                z1_array.append(z1)
                z2_array.append(z2)
                z3_array.append(z3)
        h_array = np.array(h_array)
        c_array = np.array(c_array)
        z_array = np.array(z_array)
        z0_array = np.array(z0_array)
        z1_array = np.array(z1_array)
        z2_array = np.array(z2_array)
        z3_array = np.array(z3_array)
        # q_h = find_dec_bits_kld(h_array)
        # q_c = find_dec_bits_kld(c_array)
        # q_z = find_dec_bits_kld(z_array)
        # q_z0 = find_dec_bits_kld(z0_array)
        # q_z1 = find_dec_bits_kld(z1_array)
        # q_z2 = find_dec_bits_kld(z2_array)
        # q_z3 = find_dec_bits_kld(z3_array)
        q_h = find_dec_bits_max_min(h_array)
        q_c = find_dec_bits_max_min(c_array)
        q_z = find_dec_bits_max_min(z_array)
        q_z0 = find_dec_bits_max_min(z0_array)      # not needed.
        q_z1 = find_dec_bits_max_min(z1_array)
        q_z2 = find_dec_bits_max_min(z2_array)
        q_z3 = find_dec_bits_max_min(z3_array)
        return [q_h, q_c, q_z]

    elif (type(layer.cell) is GRUCell or 'gru' in layer.cell.name):
        cfg = layer.cell.get_config()
        state = np.zeros(cfg['units'])
        k = layer.get_weights()[0]
        rk = layer.get_weights()[1]
        bias = layer.get_weights()[2]

        def gru_cell_step(cell_inputs, cell_states, kernel, recurrent_kernel, input_bias, recurrent_bias):
            h_tm1 = cell_states[0]
            # inputs projected by all gate matrices at once
            matrix_x = np.dot(cell_inputs, kernel) +  input_bias
            x_z, x_r, x_h = split_array(matrix_x, 3)
            # hidden state projected by all gate matrices at once
            matrix_inner = np.dot(h_tm1, recurrent_kernel) + recurrent_bias
            recurrent_z, recurrent_r, recurrent_h = split_array(matrix_inner, 3)
            z = nnom_sigmoid(x_z + recurrent_z)
            r = nnom_sigmoid(x_r + recurrent_r)
            hh = nnom_tanh(x_h + r * recurrent_h)
            # previous and candidate state mixed by update gate
            # h = z * h_tm1 + (1 - z) * hh
            h1 =  z*h_tm1
            h2 = 1-z
            h3 = h2 * hh
            h = h1 + h3
            return h, [h], matrix_x, matrix_inner
        h_array = []
        z_array = []
        i_array=[]
        state = [np.zeros(cfg['units'])]
        for feature in features:
            if (not layer.stateful):
                state = [np.zeros(cfg['units'])]
            for fe in feature:
                output, state, z, i = gru_cell_step(fe, state, k, rk, bias[0], bias[1])
                h_array.append(output)
                z_array.append(z)
                i_array.append(i)
        h_array = np.array(h_array)
        i_array = np.array(i_array)
        z_array = np.array(z_array)
        # q_h = find_dec_bits_kld(h_array)
        # q_i = find_dec_bits_kld(i_array)
        # q_z = find_dec_bits_kld(z_array)
        q_h = find_dec_bits_max_min(h_array)
        q_i = find_dec_bits_max_min(i_array)
        q_z = find_dec_bits_max_min(z_array)
        q_z = min(q_i, q_z)
        return [q_h, q_z]
    return []

def quantize_output(model, x_test, quantize_method='max_min', layer_offset=False, calibrate_size=None):
    # limit the test data size
    if(calibrate_size is not None):
        if (x_test.shape[0] > calibrate_size):
            x_test = x_test[:calibrate_size]
    # test, show the output ranges
    layer_q_list = {}
    # FIXME: only support one input
    if (type(model.layers[0]) != InputLayer):
        L = [model.input] + model.layers
    else:
        L = model.layers

    for layer in L:  # layer loop
        if ("input" in layer.name):
            features = x_test
        else:
            # rnn need a further step to determine the intermediate q format
            if (is_rnn_layer(layer)):
                in_layer = layer.inbound_nodes[0].inbound_layers
                layer_model = Model(inputs=model.input, outputs=in_layer.output)
                features = layer_model.predict(x_test)
                intermediate_dec = quantize_rnn_intermediate_output(layer, features)
                print(layer.name, 'dec bit', intermediate_dec)
                layer_q_list['intermediate_' + layer.name] = intermediate_dec

            # batch_normalization will need to be handled differently, since we are fusing the weight to its previosu conv.
            # sigmoid and tanh are different, their shift is fixed to 7
            if (is_shift_layer(layer) or
                    ('batch_normalization' in layer.name)):
                layer_model = Model(inputs=model.input, outputs=layer.output)
                features = layer_model.predict(x_test)
            else:
                # leave the features not changed, so this layer shift will be the same as its inputs
                pass

        # we currently only support one offset for a layer output.
        if(layer_offset):
            offset = find_offset(features)
            features = features - offset
        else:
            offset = 0
        # saturated shift using KLD method OR non saturated shift using max-min
        if ("kld"  in quantize_method
                and not is_shift_fixed(layer)
                and "input" not in layer.name
                and "dense" not in layer.name):  # test, also do not use kld in input layer
            dec_bits = find_dec_bits_kld(features, bit_width=8, scan_times=4)
            print(layer.name,"Quantized method:", "KLD", "Values max:", np.max(features), "min:", np.min(features), "dec bit", dec_bits)
        else:
            dec_bits = find_dec_bits_max_min(features, bit_width=8)
            print(layer.name,"Quantized method:","max-min"," Values max:", np.max(features), "min:", np.min(features), "dec bit", dec_bits)
        # quantise offset
        offset = int(np.round(offset * 2 ** dec_bits))
        # record the shift
        if (type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
            layer_q_list[layer.name.split(':')[0]] = [dec_bits, offset]
        else:
            layer_q_list[layer.name] = [dec_bits, offset]
        if ('batch_normalization' in layer.name):
            layer_q_list[layer.inbound_nodes[0].inbound_layers.name] = [dec_bits, offset]  # use the bn layer shift to update the last layer.

    # scan the layers backward, try to unify the dec bit in multiple input layers, (add, mult... concat...etc.)
    LM = {}
    for layer in model.layers:
        LM[layer.name] = layer
    L = [l for l in model.layers[1:]]
    L.reverse()
    def update_previous_layer_shift(layer, dec_bit):
        if(type(layer.input) == list):
            for inp in layer.input:
                iname = inp.name.split('/')[0]
                if('input' in iname):
                    continue
                layer_q_list[iname][0] = dec_min
                if(not is_shift_layer(LM[iname])):
                    update_previous_layer_shift(LM[iname], dec_bit)
        else:
            iname = layer.input.name.split('/')[0]
            if('input' in iname):
                return
            layer_q_list[iname][0] = dec_min
            if(not is_shift_layer(LM[iname])):
                update_previous_layer_shift(LM[iname], dec_bit)
    for layer in L:
        if(type(layer.input) == list):
            iname = layer.input[0].name.split('/')[0].split(':')[0]
            dec_min = layer_q_list[iname][0]
            # find min dec bit in these input
            for inp in layer.input:
                iname = inp.name.split('/')[0].split(':')[0]
                if(layer_q_list[iname][0] < dec_min):
                    dec_min = layer_q_list[iname][0]
                if(layer_q_list[iname][0] != dec_min):
                    bFlag = True
            for inp in layer.input:
                iname = inp.name.split('/')[0].split(':')[0]
                layer_q_list[iname][0] = dec_min
                if(not is_shift_layer(LM[iname])):
                    update_previous_layer_shift(LM[iname], dec_min)
            print('set dec bit', dec_min, 'for the input of', layer.name, ':', [inp.name.split('/')[0] for inp in layer.input])
            if(not is_shift_layer(layer) or dec_min < layer_q_list[layer.name][0]): # update current layer's shift only when we cannot change the shift
                layer_q_list[layer.name][0] = dec_min
    # quantise offset
    print("quantisation list", layer_q_list)
    return layer_q_list


def layer_name_from_tensor(t):
    return t.name.replace(':','/').split('/')[0]


def quantize_weights(model, name='weights.h', format='hwc', per_channel_quant=True, layer_q_list=None):
    # Quantize weights to 8-bits using (min,max) and write to file
    f = open(name, 'w')
    f.write('#include "nnom.h"\n\n')
    f.write('/* Weights, bias and Q format */\n')
    f.close()
    for curr_idx, layer in  enumerate(model.layers):
        if (not layer.weights):
            continue
        # before merging bn layer, check if the bn is "legally" after Conv
        if('batch_normalization' in layer.name) and \
            ('conv' not in layer.inbound_nodes[0].inbound_layers.name):
            raise  Exception('Only support batch_normalization placed after conv', layer.name,
                            layer.inbound_nodes[0].inbound_layers.name)
        # try to fuse BN layer to convolutional
        if ('conv' in layer.name) and \
            ('batch_normalization' in layer.outbound_nodes[0].outbound_layer.name):
            fuse_bn_to_conv(layer)
        # generate weights and bias now
        weight_dec_shift = 0
        print('quantizing weights for layer', layer.name)
        layer_weights = layer.get_weights()
        for idx, var in enumerate(layer_weights):
            var_name = convert_tensor_name(layer.weights[idx])
            var_values = var
            if("kernel" not in var_name and 'bias' not in var_name): # ignore batchnormalisation's parameters
                continue

            if (per_channel_quant and type(layer) in [Conv2D, Conv1D, DepthwiseConv2D, Conv2DTranspose]):
                if(type(layer) in [DepthwiseConv2D] and "kernel" in var_name): #depthwise kernel quantised by
                    shape = var_values.shape[:2] + (-1,) # need to combine the mult and channel first
                    var = var_values.reshape(shape)
                    dec_bits = find_dec_bits_max_min_axis(var, axis=-1, bit_width=8)
                elif(type(layer) in [Conv2DTranspose]):
                    dec_bits = find_dec_bits_max_min_axis(var_values, axis=-2, bit_width=8)
                else:
                    dec_bits = find_dec_bits_max_min_axis(var_values, bit_width=8)
            else:
                dec_bits = find_dec_bits_max_min(var_values, bit_width=8)
            print('   ', var_name, "dec bit", dec_bits)

            # kernel dec, bias dec, bias shift, output shift
            if(is_shift_layer(layer) and not is_rnn_layer(layer)):
                inp = layer.input.name.replace(':','/').split('/')[0]
                layer_input_dec = layer_q_list[inp][0]
                layer_output_dec = layer_q_list[layer.name][0]
                if ("kernel" in var_name):
                    weight_dec_shift = dec_bits
                else:
                    # channel wise
                    if hasattr(dec_bits, '__len__'):
                        bias_shift = np.full(len(dec_bits), layer_input_dec)+weight_dec_shift-dec_bits
                        layer_output_shift = np.full(len(weight_dec_shift), layer_input_dec) + weight_dec_shift \
                            - np.full(len(weight_dec_shift), layer_output_dec)
                        if (np.min(bias_shift) < 0):
                            for i, w_dec in enumerate(weight_dec_shift):
                                if (bias_shift[i] < 0):
                                    dec_bits[i] = w_dec
                                    bias_shift[i] = 0
                    # layer wise
                    else:
                        bias_shift = layer_input_dec + weight_dec_shift - dec_bits
                        layer_output_shift = layer_input_dec + weight_dec_shift - layer_output_dec
                        if (bias_shift < 0):
                            dec_bits = weight_dec_shift
                            bias_shift = 0
            # RNN layer's kernel dec, bias dec, bias shift, output shift
            if(is_rnn_layer(layer)):
                inp = layer.input.name.replace(':','/').split('/')[0]
                layer_input_dec = layer_q_list[inp][0]
                layer_output_dec = layer_q_list[layer.name][0]
                #if (type(layer.cell) is SimpleRNNCell):
                if ("kernel" in var_name and 'recurrent' not in var_name):
                    weight_dec_shift = dec_bits
                elif ('bias' in var_name):
                    bias_shift = layer_input_dec + weight_dec_shift - dec_bits
                    layer_output_shift = layer_input_dec + weight_dec_shift - layer_output_dec # this is not valid
                    if (bias_shift < 0):
                        dec_bits = weight_dec_shift
                        bias_shift = 0

            # now quantise them
            if(type(layer) in [Conv2D, Conv1D, DepthwiseConv2D, Conv2DTranspose]):
                if(type(layer) in [DepthwiseConv2D] and "kernel" in var_name):
                    old_shape = var_values.shape
                    var_values = quantize_data(var_values.reshape(var_values.shape[:2] + (-1,)),
                                   dec_bits, axis=-1, per_axis=per_channel_quant) # convert to [h, w, out x mult]
                    var_values = var_values.reshape(old_shape) # convert the shape back to  [h, w, out, mult]
                elif(type(layer) in [Conv2DTranspose] and "kernel" in var_name):
                    var_values = quantize_data(var_values, dec_bits, axis=-2, per_axis=per_channel_quant) # [h, w, out, in]
                else:
                    var_values = quantize_data(var_values, dec_bits, per_axis=per_channel_quant) # [h, w, in, out]
            else:
                var_values = quantize_data(var_values, dec_bits, per_axis=False)

            # CHW format
            if ('chw' in format):
                if (is_lstm_layer(layer) or is_gru_layer(layer)):   # currently we use 16 bit intermediate， use reorder optimation
                    transposed_wts = np.transpose(var_values)
                    if('kernel' in var_name):
                        transposed_wts = convert_q7_q15_weights(np.reshape(transposed_wts ,(transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
                # dense and rnn still working under HWC format
                elif ("dense" in var_name or is_rnn_layer(layer)) and "kernel" in var_name:
                    transposed_wts = np.transpose(var_values)
                    transposed_wts = convert_to_x4_q7_weights(np.reshape(transposed_wts, (transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
                # all other kernels, bias stay the same
                else:
                    transposed_wts = var_values
            # HWC format (NNOM/CMSIS-NN use [out_ch, h, w, in_ch], in C order)
            else:
                if (len(var_values.shape) == 3):  # 1D convolution layer weights
                    transposed_wts = np.transpose(var_values, (2, 0, 1))
                elif (len(var_values.shape) == 4):  # 2D convolution layer weights
                    if(type(layer) == Conv2DTranspose): # test
                        transposed_wts = np.transpose(var_values, (2, 0, 1, 3))
                    elif type(layer) == DepthwiseConv2D:
                        transposed_wts = var_values#np.transpose(var_values, (0, 1, 3, 2)) # [h, w, out, mult] test for multiplier
                    else:
                        transposed_wts = np.transpose(var_values, (3, 0, 1, 2))
                elif(is_lstm_layer(layer) or is_gru_layer(layer)):   # currently we use 16 bit intermediate, use reorder optimation
                    if('kernel' in var_name):
                        transposed_wts = np.transpose(var_values)
                        transposed_wts = convert_q7_q15_weights(np.reshape(transposed_wts ,(transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
                    else: # bias will not need to be transposed (for GRU which has 2d bias)
                        transposed_wts = var_values
                else:  # fully connected layer weights or biases of any layer
                    # test, use opt weight reorder
                    transposed_wts = np.transpose(var_values)
                    if ("dense" in var_name or is_rnn_layer(layer)) and "kernel" in var_name: # and other RNN layers
                        transposed_wts = convert_to_x4_q7_weights(np.reshape(transposed_wts ,(transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))

            with open(name, 'a') as f:
                def write_weights(f, name, value):
                    f.write('#define ' + name + ' {')
                    value.tofile(f, sep=", ", format="%d")
                    f.write('}\n\n')
                # weights or bias
                write_weights(f, var_name.upper(), transposed_wts)
                # dec bits
                write_weights(f, var_name.upper()+'_DEC_BITS' , np.array(dec_bits))
                # for test
                if( "bias" in var_name):
                    f.write('#define ' + layer.name.upper() + '_BIAS_LSHIFT '+to_cstyle(bias_shift) +'\n\n')
                    #f.write('#define ' + layer.name.upper() + '_OUTPUT_DEC '+ to_cstyle(layer_output_dec)+'\n\n') # not here
                    f.write('#define ' + layer.name.upper() + '_OUTPUT_RSHIFT ' + to_cstyle(layer_output_shift)+'\n\n')


def generate_model(model, x_test, per_channel_quant=False, name='weights.h', format='hwc', quantize_method='max_min'):
    """
    :param model:
    :param x_test:
    :param name:
    :param format:
    :param quantize_method: "max_min" or "kld"
    :return:
    """
    # get the quantize output range/format
    layer_q_list = quantize_output(model, x_test, layer_offset=False, quantize_method=quantize_method)
    # quantize weights and output shift
    quantize_weights(model, per_channel_quant=per_channel_quant, name=name, format=format, layer_q_list=layer_q_list)
    # now generate the model
    if (type(model.layers[0]) != InputLayer):
        L = [model.input] + model.layers
    else:
        L = model.layers
    with open(name, 'a') as fp:
        # generate the list of output
        fp.write('\n/* output q format for each layer */\n')
        for layer in L:
            if (type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
                iname = layer.name.split(':')[0]
            else:
                iname = layer.name
            fp.write('#define %s_OUTPUT_DEC %s\n' % (iname.upper(), layer_q_list[iname][0]))
            fp.write('#define %s_OUTPUT_OFFSET %s\n' % (iname.upper(), layer_q_list[iname][1]))
        fp.write('\n/* bias shift and output shift for none-weighted layer */\n')

        # generate output shift for the layers without weights (weighted layers were generated in quantize_weights)
        for layer in model.layers:
            if (is_shift_layer(layer)):
                iname = layer.name.upper()
                # add, sub
                if ('add' in layer.name or 'subtract' in layer.name):
                    # only consider the first, they have been set to same in out_put_range()
                    inp = layer.input[0].name.replace(':', '/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_DEC-{0}_OUTPUT_DEC)\n'.format(
                        iname, inp))
                    fp.write(
                        '#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(
                            iname))
                # mult is different, Q3.4 * Q3.4 = Q6.8. if mult out is Q4.3, then shift (Q.4+q.4)-Q.3=5. Am I right?
                elif ('multiply' in layer.name):
                    inp = layer.input[0].name.replace(':', '/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_DEC*2-{0}_OUTPUT_DEC)\n'.format(
                        iname, inp))
                    fp.write(
                        '#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(
                            iname))

        fp.write('\n/* tensors and configurations for each layer */\n')
        LI = {}
        ID = 0

        def is_skipable_layer(layer):
            # FIXME: add more that could be skiped
            if ('lambda' in layer.name or
                'dropout' in layer.name or
                'batch_normalization' in layer.name
                #or ('flatten' in layer.name and 'chw' not in format)
                ): # flatten layer can be skipped in HWC but needed in CHW
                return True
            return False

        output_num = 0
        for id, layer in enumerate(L):
            if (is_skipable_layer(layer)):
                inp = layer.input.name.replace(':', '/').split('/')[0]
                LI[layer.name] = (LI[inp][0], layer)
            else:
                if (type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
                    LI[layer.name.split(':')[0]] = (ID, layer)
                else:
                    LI[layer.name] = (ID, layer)
                ID += 1

            def gen_weight_tensor(w, per_axis):
                var_cname = convert_tensor_name(w) + '_data'
                dec_bits_name = convert_tensor_name(w).upper() + '_DEC_BITS'
                fp.write(gen_values(var_cname, convert_tensor_name(w).upper()))
                fp.write(gen_tensor(w, dec_bits=dec_bits_name, tensor_value=var_cname, per_axis=per_axis))

            # output the config of all layer
            if (type(layer) in [InputLayer] or 'input' in layer.name):
                if(type(layer) == tf.Tensor):
                    raise  Exception('Not yet support tensor as input/or Sequential model. '
                                     'please use Input layer as your first layer in the model', layer.name, layer)
                size = 1
                for s in layer.input.shape[1:]:
                    size *= s if s is not None else 1
                fp.write(gen_values('nnom_input_data', '{0}', size=str(size), dtype='static int8_t'))
                fp.write(gen_tensor(layer.input, layer_q_list[layer.name][0], tensor_value='nnom_input_data', is_io_tensor=True))
                fp.write(gen_io_config(layer, tensor_name=convert_tensor_name(layer.input)))
            elif (type(layer) in [Conv2D, Conv1D, DepthwiseConv2D]):
                for w in layer.weights:
                    gen_weight_tensor(w, per_axis=per_channel_quant)
                fp.write(gen_conv2d_config(layer, layer.name.upper() +'_OUTPUT_RSHIFT', layer.name.upper() +'_BIAS_LSHIFT'))
            elif (type(layer) in [Conv2DTranspose]):
                for w in layer.weights:
                    gen_weight_tensor(w, per_axis=per_channel_quant)
                fp.write(gen_conv2d_trans_config(layer, layer.name.upper() +'_OUTPUT_RSHIFT', layer.name.upper() +'_BIAS_LSHIFT'))
            elif (type(layer) in [Dense]):
                for w in layer.weights:
                    gen_weight_tensor(w, per_axis=False)
                fp.write(gen_dense_config(layer, layer.name.upper() +'_OUTPUT_RSHIFT', layer.name.upper() +'_BIAS_LSHIFT'))
            elif (type(layer) in [MaxPooling2D, AveragePooling2D, MaxPooling1D, AveragePooling1D]):
                fp.write(gen_pooling_config(layer))
            elif (type(layer) in [GlobalMaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D]):
                fp.write(gen_gl_pooling_config(layer))
            elif (type(layer) in [Multiply, Add, Subtract]):
                fp.write(gen_matrix_config(layer, output_shift_name=layer.name.upper()+'_OUTPUT_RSHIFT'))
            elif (type(layer) in [ZeroPadding2D, ZeroPadding1D]):
                fp.write(gen_zero_padding_config(layer))
            elif (type(layer) in [Cropping2D, Cropping1D]):
                fp.write(gen_cropping_config(layer))
            elif (type(layer) in [Softmax]):
                fp.write(gen_softmax_config(layer))
            elif (type(layer) in [Flatten]):
                fp.write(gen_flatten_config(layer))
            elif (type(layer) in [Concatenate]):
                fp.write(gen_concat_config(layer))
            elif (type(layer) in [Lambda]):
                fp.write(gen_lambda_config(layer))
            elif (type(layer) in [UpSampling2D, UpSampling1D]):
                fp.write(gen_upsampling_config(layer))
            elif(is_rnn_layer(layer)):
                if(type(layer.cell) is SimpleRNNCell):
                    for w in layer.weights:
                        gen_weight_tensor(w, per_axis=False)
                    fp.write(gen_simple_cell_config(layer, layer_q_list['intermediate_'+layer.name]))
                elif(type(layer.cell) is GRUCell or 'gru' in layer.cell.name):
                    for w in layer.weights:
                        gen_weight_tensor(w, per_axis=False)
                    fp.write(gen_gru_cell_config(layer, layer_q_list['intermediate_'+layer.name]))
                elif(type(layer.cell) is LSTMCell or 'lstm' in layer.cell.name):
                    for w in layer.weights:
                        gen_weight_tensor(w, per_axis=False)
                    fp.write(gen_lstm_cell_config(layer, layer_q_list['intermediate_'+layer.name]))
                fp.write(gen_rnn_config(layer))

            # test, multiple output layer
            if(len(layer.outbound_nodes) == 0):
                size=1
                for s in layer.output.shape[1:]:
                    size *= s if s is not None else 1
                if(output_num == 0): # the first output or the only output
                    fp.write(gen_values('nnom_output_data', '{0}', size=str(size), dtype='static int8_t'))
                    fp.write(gen_output_config(layer, dec_bits=layer.name.upper() + '_OUTPUT_DEC', output_num=output_num, value_name='nnom_output_data'))
                    output_num += 1
                else:
                    output_value_names = 'nnom_output_data'+str(output_num)
                    fp.write(gen_values(output_value_names, '{0}', size=str(size), dtype='static int8_t'))
                    fp.write(gen_output_config(layer, dec_bits=layer.name.upper() + '_OUTPUT_DEC', output_num=output_num, value_name=output_value_names))
                    output_num += 1

            # # last layer, attach the additional nnom output layer
            # if(id == len(L)-1):
            #     size=1
            #     for s in layer.output.shape[1:]:
            #         size *= s if s is not None else 1
            #     fp.write(gen_values('nnom_output_data', '{0}', size=str(size), dtype='static int8_t'))
            #     fp.write(gen_output_config(layer,  dec_bits=layer.name.upper()+'_OUTPUT_DEC', value_name='nnom_output_data'))

        # write version
        fp.write('/* model version */\n')
        fp.write('#define NNOM_MODEL_VERSION (10000*{0} + 100*{1} + {2})\n'.format(model_major_version, model_sub_version, model_reversion ))

        # model
        fp.write('\n/* nnom model */\n')
        fp.write('static nnom_model_t* nnom_model_create(void)\n{\n')
        fp.write('\tstatic nnom_model_t model;\n')
        if (ID > 32):
            fp.write('\tnnom_layer_t ** layer = malloc(sizeof(nnom_layer_t *)*%d);\n' % (ID + 1))
            fp.write('\tif(NULL == layer) return NULL;\n')
        else:
            fp.write('\tnnom_layer_t* layer[%d];\n' % (ID + 1))
        fp.write('\n\tcheck_model_version(NNOM_MODEL_VERSION);')
        fp.write('\n\tnew_model(&model);\n\n')

        # inverted order of output, very strange
        output_num = (len(model.output) -1) if type(model.output) is list else 0
        for layer in L:
            if (is_skipable_layer(layer)):
                continue
            # FIXME: need a better solution to seperate the input 'tensor' from other layers
            if (type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
                id, _ = LI[layer.name.split(':')[0]]
            else:
                id, _ = LI[layer.name]

            if ('input' in layer.name):
                fp.write('\tlayer[%d] = input_s(&%s_config);\n' % (id, layer.name))

            # convlutional
            elif ('conv1d' in layer.name
                  or 'conv2d' in layer.name):
                inp = layer_name_from_tensor(layer.input)
                if('transpose' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(conv2d_trans_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name,  LI[inp][0]))
                elif('depthwise' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(dw_conv2d_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))
                else:
                    fp.write('\tlayer[{0}] = model.hook(conv2d_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))
            elif ('activation' in layer.name):
                inp = layer_name_from_tensor(layer.input)
                cfg = layer.get_config()
                if (cfg['activation'] == 'relu'):
                    fp.write('\tlayer[%s] = model.active(act_relu(), layer[%s]);\n' % (id, LI[inp][0]))
                elif (cfg['activation'] == 'tanh'):
                    fp.write('\tlayer[%s] = model.active(act_hard_tanh(%s_OUTPUT_DEC), layer[%s]);\n' % (
                    id, inp.upper(), LI[inp][0]))
                elif (cfg['activation'] == 'sigmoid'):
                    fp.write('\tlayer[%s] = model.active(act_sigmoid(%s_OUTPUT_DEC), layer[%s]);\n' % (
                    id, inp.upper(), LI[inp][0]))
                elif (cfg['activation'] == 'hard_sigmoid'):
                    fp.write('\tlayer[%s] = model.active(act_hard_sigmoid(%s_OUTPUT_DEC), layer[%s]);\n' % (
                    id, inp.upper(), LI[inp][0]))
                elif (cfg['activation'] == 'softmax'):
                    fp.write('\tlayer[%s] = model.hook(Softmax(), layer[%s]);\n' % (id, LI[inp][0]))
            elif ('leaky_re_lu' in layer.name):
                inp = layer_name_from_tensor(layer.input)
                cfg = layer.get_config()
                fp.write('\tlayer[%s] = model.active(act_leaky_relu(%ff), layer[%s]);\n' % (id, cfg["alpha"],LI[inp][0]))
            elif ('re_lu' in layer.name):
                inp = layer_name_from_tensor(layer.input)
                cfg = layer.get_config()
                if(cfg['max_value'] is None and cfg['negative_slope'] == 0 and cfg['threshold'] == 0):
                    fp.write('\tlayer[%s] = model.active(act_relu(), layer[%s]);\n' % (id, LI[inp][0]))
                else:
                    if(cfg['max_value'] is None):
                        max_v = 'INFINITY '
                    else:
                        max_v = str(cfg['max_value'])
                    fp.write('\tlayer[%s] = model.active(act_adv_relu(%f,%s,%f), layer[%s]);\n'
                             % (id, cfg['negative_slope'], max_v, cfg['threshold'], LI[inp][0]))
            # pooling
            elif ('max_pooling' in layer.name):
                inp = layer_name_from_tensor(layer.input)
                if ('global' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(global_maxpool_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))
                else:
                    fp.write('\tlayer[{0}] = model.hook(maxpool_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))
            elif ('average_pooling' in layer.name):
                inp = layer_name_from_tensor(layer.input)
                if ('global' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(global_avgpool_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))
                else:
                    fp.write('\tlayer[{0}] = model.hook(avgpool_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))
            elif ('up_sampling' in layer.name):
                inp = layer_name_from_tensor(layer.input)
                fp.write('\tlayer[{0}] = model.hook(upsample_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))
            # zero padding
            elif ('zero_padding' in layer.name):
                inp = layer_name_from_tensor(layer.input)
                fp.write('\tlayer[{0}] = model.hook(zeropadding_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))
            # Cropping
            elif ('cropping' in layer.name):
                inp = layer_name_from_tensor(layer.input)
                fp.write('\tlayer[{0}] = model.hook(cropping_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))

            # others
            elif ('flatten' in layer.name):  # flatten is needed in CHW backend but not needed in HWC
                inp = layer_name_from_tensor(layer.input)
                fp.write('\tlayer[{0}] = model.hook(flatten_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))
            elif ('concatenate' in layer.name):
                inps = [layer_name_from_tensor(input) for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]' % (LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(concat_s(&%s_config), %s%s);\n' % (
                    id, layer.name, len(inps), inX))
            elif ('add' in layer.name):
                inps = [layer_name_from_tensor(input) for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]' % (LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(add_s(&%s_config), %s%s);\n' % (
                    id, layer.name, len(inps), inX))
            elif ('subtract' in layer.name):
                inps = [layer_name_from_tensor(input) for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]' % (LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(sub_s(&%s_config), %s%s);\n' % (
                    id, layer.name, len(inps), inX))
            elif ('multiply' in layer.name):
                inps = [layer_name_from_tensor(input) for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]' % (LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(mult_s(&%s_config), %s%s);\n' % (
                    id, layer.name, len(inps), inX))
            elif ('dense' in layer.name):
                inp = layer_name_from_tensor(layer.input)
                fp.write('\tlayer[{0}] = model.hook(dense_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))
            elif ('softmax' in layer.name):
                inp = layer_name_from_tensor(layer.input)
                fp.write('\tlayer[{0}] = model.hook(softmax_s(&{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0]))

            elif (is_rnn_layer(layer)):
                inp = layer_name_from_tensor(layer.input)
                line = '\tlayer[{0}] = model.hook(rnn_s(<rnn_cell>, &{1}_config), layer[{2}]);\n'.format(id, layer.name, LI[inp][0])
                if (type(layer.cell) is SimpleRNNCell):
                    line = line.replace('<rnn_cell>', 'simple_cell_s(&%s_simple_cell_config)' %(layer.name))
                elif (type(layer.cell) is GRUCell or 'gru' in layer.cell.name):
                    line = line.replace('<rnn_cell>', 'gru_cell_s(&%s_gru_cell_config)' % (layer.name))
                elif (type(layer.cell) is LSTMCell or 'lstm' in layer.cell.name):
                    line = line.replace('<rnn_cell>', 'lstm_cell_s(&%s_lstm_cell_config)' % (layer.name))
                fp.write(line)
            else:
                raise Exception('unsupported layer', layer.name, layer)

            # test, multiple output layer (not yet working with multiple outputs)
            if(len(layer.outbound_nodes) == 0):
                fp.write('\tlayer[{0}] = model.hook(output_s(&{1}_config), layer[{2}]);\n'.format(id + 1, 'output'+str(output_num), LI[inp][0] + 1))
                output_num -=1 # the num is inverted in keras, not a good solution yet.

            """
            # temporary fixed for activations attached into layers in construction
            def is_activation_attached(layer):
                if(("Softmax" in layer.output.name and "softmax" not in layer.name)or
                ("Relu" in layer.output.name and "re_lu" not in layer.name) or
                ("Sigmoid" in layer.output.name and "sigmoid" not in layer.name) or
                ("Tanh" in layer.output.name and "tanh" not in layer.name)):
                    return True
                return False
            if "input" not in layer.name and is_activation_attached(layer):
                inp = layer.output.name.replace(':', '/').split('/')[0]
                cfg = layer.get_config()
                if(cfg['activation'] == 'relu'):
                    fp.write('\tlayer[%s] = model.active(act_relu(), layer[%s]);\n'%(id, LI[inp][0]))
                if(cfg['activation'] == 'tanh'):
                    fp.write('\tlayer[%s] = model.active(act_tanh(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
                if(cfg['activation'] == 'sigmoid'):
                    fp.write('\tlayer[%s] = model.active(act_sigmoid(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
                elif(cfg['activation'] == 'softmax'):
                    fp.write('\tlayer[%s] = model.hook(Softmax(), layer[%s]);\n'%(id, LI[inp][0]))
            """
        # generate final output layer
        #fp.write('\tlayer[{0}] = model.hook(output_s(&{1}_config), layer[{2}]);\n'.format(id+1, 'output', LI[inp][0]+1))
        fp.write('\tmodel_compile(&model, layer[0], layer[%s]);\n' % (id + 1))
        if (ID > 32):
            fp.write('\tfree(layer);\n')
        fp.write('\treturn &model;\n}\n')
    with open('.layer_q_list', 'w') as fp:
        fp.write(str(layer_q_list))

def evaluate_model(model, x_test, y_test, running_time=False, to_file='evaluation.txt'):
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', scores[0])
    print('Top 1:', scores[1])

    if(len(y_test.shape)>1):
        predictions = model.predict(x_test)
        matrix = skmetrics.confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
        print(matrix)

    run_time = 0
    if running_time:
        # try to calculate the time
        T = time.time()
        for i in range(10):
            model.predict(x_test)
        T = time.time() - T
        run_time = round((T / 10 / x_test.shape[0] * 1000 * 1000), 2)
        print("Runing time:",run_time , "us" )
    #
    with open(to_file, 'w') as f:
        f.write("Runing time: "+ str(run_time) + "us" + "\n")
        f.write('Test loss:'+ str(scores[0]) + "\n")
        f.write('Top 1:'+ str(scores[1])+ "\n")
        if (len(y_test.shape) > 1):
            for row in matrix:
                row.tofile(f, sep=',')
                f.write("\n")
    return scores

def f2q(d, Q):
    '''To convert a number from floating point to Qm.n format:
        1. Multiply the floating point number by 2n
        2. Round to the nearest integer
    '''
    return np.round(d*2**Q)


def q2f(d, Q):
    '''To convert a number from Qm.n format to floating point:
        1. Convert the number to floating point as if it were an integer, in other words remove the binary point
        2. Multiply by 2-n
    '''
    return d*2**-Q

def show_weights(w, name):
    sz = 1
    for s in w.shape:
        sz = sz*s
    aL = w.reshape(sz,)
    MIN,MAX=min(aL),max(aL)
    Q = int(np.ceil(np.log2(max(abs(MIN),abs(MAX)))))
    Q = 7-Q
    qL = f2q(aL,Q)
    qL = q2f(qL,Q)
    plt.figure(figsize=(18, 3))
    plt.subplot(131)
    plt.title(name)
    plt.plot(aL)
    plt.grid()
    aL.sort()
    plt.plot(aL,'r')
    plt.grid()
    plt.subplot(132)
    plt.title('Q%s'%(Q))
    qL.sort()
    plt.plot(aL,'r')
    plt.plot(qL,'g')
    plt.grid()
    plt.subplot(133)
    plt.hist(aL,100)
    plt.title('hist')
    plt.grid()
    plt.show()

def compare(a,b,name):
    sz = 1
    for s in a.shape:
        sz = sz*s
    aL = a.reshape(sz,)
    bL = b.reshape(sz,)
    assert(len(aL) == len(bL))
    Z = list(zip(aL,bL))
    Z.sort(key=lambda x: x[0])
    aL1,bL1=zip(*Z)
    plt.figure(figsize=(18, 3))
    plt.subplot(131)
    plt.plot(aL)
    plt.plot(aL1,'r')
    plt.grid()
    plt.title('tf-%s'%(name))
    plt.subplot(133)
    plt.plot(bL1,'g')
    plt.plot(aL1,'r')
    plt.grid()
    plt.title('compare')
    plt.subplot(132)
    bL1=list(bL1)
    bL1.sort()
    plt.plot(bL)
    plt.plot(bL1,'g')
    plt.grid()
    plt.title('nn-%s'%(name))
    plt.show()

