'''
    Copyright (c) 2018-2019
    Jianjia Ma, Wearable Bio-Robotics Group (WBR)
    majianjia@live.com

    SPDX-License-Identifier: Apache-2.0

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
import tensorflow as tf
from keras.engine.input_layer import InputLayer
from keras.layers import Lambda, BatchNormalization
from keras.models import Model
from keras import backend as K
from sklearn import metrics
from fully_connected_opt_weight_generation import *
import time
import warnings


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

def fake_clip_min_max(min=0, max =1, bit=8):
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
    # quantize input x
    min_value = np.min(x)
    max_value = np.max(x)

    int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
    dec_bits = 7 - int_bits
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
       ('activation' in layer.name and layer.get_config()['activation'] == 'sigmoid') or
       ('activation' in layer.name and layer.get_config()['activation'] == 'tanh')
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
        ('activation' in layer.name and layer.get_config()['activation'] == 'tanh')
    ):
        return True
    return  False

def fuse_bn_to_conv(layer):
    # try to fuse BN layer to convolutional
    if ('conv' in layer.name) and \
            ('batch_normalization' in layer._outbound_nodes[0].outbound_layer.name):

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

        if ('conv2d' in layer.name):
            epsilon = 1e-3  # default epsilon for tf.slim.batch_norm
            for l in range(c_w.shape[3]):
                for k in range(c_w.shape[2]):
                    for j in range(c_w.shape[1]):
                        for i in range(c_w.shape[0]):
                            if "depthwise" in layer.name:  # depthwise batchnorm params are ordered differently
                                c_w[i][j][k][l] *= bn_gamma[k] / np.sqrt(bn_variance[k] + epsilon)
                            else:
                                c_w[i][j][k][l] *= bn_gamma[l] / np.sqrt(bn_variance[l] + epsilon)

            if "depthwise" in layer.name:
                depth_dim = c_w.shape[2]
            else:
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
                depth_dim = c_w.shape[1]
            else:
                depth_dim = c_w.shape[2]
            for l in range(depth_dim):
                c_b[l] = (bn_gamma[l] * (c_b[l] - bn_mean[l]) / np.sqrt(bn_variance[l] + epsilon)) + bn_beta[l]

        print('fused weight max', c_w.max(), 'min', c_w.min())
        print('fused bias max', c_b.max(), 'min', c_b.min())
        # write the weights back to the layer
        # after that, the model will be destroyed.. need a better way to pass the new weight
        layer.set_weights([c_w, c_b])

def generate_weights(model, name='weights.h', format='hwc', shift_list=None):
    # Quantize weights to 8-bits using (min,max) and write to file
    f = open(name, 'w')
    f.write('#include "nnom.h"\n\n')
    f.close()

    for curr_idx, layer in  enumerate(model.layers):
        if (not layer.weights):
            continue

        # before merging bn layer, check if the bn is "legally" after Conv
        if('batch_normalization' in layer.name) and \
            ('conv' not in layer._inbound_nodes[0].inbound_layers[0].name):
            raise  Exception('Currently only support batch_normalization after conv', layer.name,
                            layer._inbound_nodes[0].inbound_layers[0].name)

        # try to fuse BN layer to convolutional
        if ('conv' in layer.name) and \
            ('batch_normalization' in layer._outbound_nodes[0].outbound_layer.name):
            fuse_bn_to_conv(layer)

        # generate weights and bias now
        weight_dec_shift = 0
        print('weights for layer', layer.name)
        for var in layer.weights:
            var_name = str(var.name)
            if("kernel" in var_name ):
                var_values = layer.get_weights()[0] # weight
                print("  weight:", var_name)
            elif("bias" in var_name):
                var_values = layer.get_weights()[1] # bias
                print("  bias: ",var_name)
            else:
                continue

            print("  original shape: ", var_values.shape)
            min_value = np.min(var_values)
            max_value = np.max(var_values)

            int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
            dec_bits = 7 - int_bits
            print("  dec bit", dec_bits)
            bSameAsKernel = False
            if(is_shift_layer(layer)):
                bSameAsKernel = False
                inp = layer.input.name.replace(':','/').split('/')[0]
                input_encoding = shift_list[inp]
                if ("kernel" in var_name):
                    weight_dec_shift = dec_bits
                else:
                    shift = input_encoding+weight_dec_shift-dec_bits
                    if(shift < 0):
                        bSameAsKernel = True
            if(shift_list is None or bSameAsKernel):
                # check if bias shift > weight shift, then reduce bias shift to weight shift	
                if ("kernel" in var_name):
                    weight_dec_shift = dec_bits	
                else:	
                    if(dec_bits > weight_dec_shift):	
                        dec_bits = weight_dec_shift	
                print("  new dec bit", dec_bits)

            # convert to [-128,128) or int8
            var_values = np.round(var_values * 2 ** dec_bits)
            var_name = var_name.replace('/', '_')
            var_name = var_name.replace(':', '_')
            with open(name, 'a') as f:
                f.write('#define ' + var_name.upper() + ' {')
            # CHW format
            if ('chw' in format):
                if "dense" in var_name and "kernel" in var_name:
                    transposed_wts = np.transpose(var_values)
                    transposed_wts = convert_to_x4_q7_weights(
                        np.reshape(transposed_wts, (transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
                # all other kernels, bias stay the same
                else:
                    transposed_wts = var_values
            # HWC format
            else:
                if (len(var_values.shape) == 3):  # 1D convolution layer weights
                    transposed_wts = np.transpose(var_values, (2, 0, 1))
                elif (len(var_values.shape) == 4):  # 2D convolution layer weights
                    transposed_wts = np.transpose(var_values, (3, 0, 1, 2))
                else:  # fully connected layer weights or biases of any layer
                    # test, use opt weight reorder
                    if "dense" in var_name and "kernel" in var_name:
                        transposed_wts = np.transpose(var_values)
                        transposed_wts = convert_to_x4_q7_weights(np.reshape(transposed_wts ,(transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
                    else:
                        transposed_wts = np.transpose(var_values)

            print("  reshape to:",transposed_wts.shape)

            with open(name, 'a') as f:
                transposed_wts.tofile(f, sep=", ", format="%d")
                f.write('}\n\n')
                if ("bias" in var_name):
                    f.write('#define ' + var_name.upper() + '_SHIFT ' + '(' + str(dec_bits) + ')\n\n\n')
                if ("kernel" in var_name ):
                    f.write('#define ' + var_name.upper() + '_SHIFT ' + '(' + str(dec_bits) + ')\n\n')
            """
            # for checking the quantised and dequantised range. 
            with K.tf.Session() as session:
                # convert back original range but quantized to 8-bits or 256 levels
                var_values = var_values / (2 ** dec_bits)
                var_values = session.run(K.tf.assign(var, var_values))
                print('  '+var_name + ' number of wts/bias: ' + str(var_values.shape) + \
                  ' dec bits: ' + str(dec_bits) + \
                  ' max: (' + str(np.max(var_values)) + ',' + str(max_value) + ')' + \
                  ' min: (' + str(np.min(var_values)) + ',' + str(min_value) + ')')
            """

def layers_output_ranges(model, x_test, kld=True, calibrate_size=1000):
    # limit the test data size
    np.random.shuffle(x_test)
    if(x_test.shape[0] > calibrate_size):
        x_test = x_test[:1000]
    # test, show the output ranges
    shift_list = {}
    # FIXME: only support one input
    if(type(model.layers[0]) != InputLayer):
        L = [model.input] + model.layers
    else:
        L = model.layers
    last_layer = None

    for layer in L: # layer loop
        if("input" in layer.name):
            features = x_test
        else:
            # batch_normalization will need to be handled differently, since we are fusing the weight to its predecessor.
            # sigmoid and tanh are different, their shift is fixed to 7
            if(is_shift_layer(layer) or
                ('batch_normalization' in layer.name)):
                layer_model = Model(inputs=model.input, outputs=layer.output)
                features = layer_model.predict(x_test)
            else:
                # leave the features not changed, so this layer shift will be the same
                # as its inputs
                pass
        #  calculate no saturation shift
        max_val = features.max()
        min_val = features.min()
        int_bits = int(np.ceil(np.log2(max(abs(max_val), abs(min_val)))))
        dec_bits = 7 - int_bits

        # saturation shift, using KLD method
        # Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        if(kld and not is_shift_fixed(layer) and "input" not in layer.name and "dense" not in layer.name): # test, also do not use kld in input layer
            import scipy.stats
            abs_max = max(abs(max_val), abs(min_val))
            small_var = 1e-5
            bins = np.arange(-abs_max, abs_max, abs_max/2048*2)
            q_bins = np.arange(-abs_max, abs_max, abs_max/256*2)
            flat_hist = np.histogram(features.flatten(), bins=bins)[0]
            kl_loss = []
            kl_shifts = []
            for shift in range(4):
                t = 2 ** (dec_bits + shift)     # 2-based threshold
                act = np.round(features.flatten() * t)
                act = act / t
                act = np.clip(act, -128/t, 127/t)
                act = np.histogram(act, bins=q_bins)[0]
                act_hist = np.zeros(2047)
                chunk = int(2048/256)
                for i in range(int(255)):
                    none_zero = np.count_nonzero(flat_hist[i*chunk:(i+1)*chunk])
                    if none_zero == 0:
                        continue
                    for j in range(chunk):
                        act_hist[i*chunk+j] = act[i]/none_zero if flat_hist[i*chunk+j] != 0 else 0
                flat_hist[flat_hist==0] = small_var
                act_hist[act_hist==0] = small_var
                kl = scipy.stats.entropy(flat_hist, act_hist)
                kl_loss.append(kl)
                kl_shifts.append(dec_bits + shift)
                """
                ax = plt.subplot(8, 1, shift+1)
                ax.plot(flat_hist)
                ax.plot(act_hist)
                """
            new_dec = kl_shifts[np.argmin(kl_loss)] # set the dec_bit to the KLD results
            #plt.show()
            print("KLD loss", kl_loss)
            print("KLD shift", kl_shifts)
            if(new_dec != dec_bits):
                print(layer.name,"is using KLD method, original shift",dec_bits, "KLD results", new_dec)
                dec_bits = new_dec

        print( layer.name, "max value:", max_val, "min value:", min_val,"dec bit", dec_bits)
        # record the shift
        if(model.input == layer and type(model.layers[0]) != InputLayer):
            shift_list[layer.name.split(':')[0]] = dec_bits
        else:
            shift_list[layer.name] = dec_bits
        if ('batch_normalization' in layer.name):
            shift_list[last_layer.name] = dec_bits  # use the bn layer shift to update the last layer.
        last_layer = layer

    LM = {}
    for layer in model.layers:
        LM[layer.name] = layer
    L = [l for l in model.layers[1:]]
    L.reverse()

    def update_previous_layer_shift(layer, Q):
        if(type(layer.input) == list):
            for inp in layer.input:
                iname = inp.name.split('/')[0]
                if('input' in iname):
                    continue
                shift_list[iname] = Qmin
                if(not is_shift_layer(LM[iname])):
                    update_previous_layer_shift(LM[iname], Q)
        else:
            iname = layer.input.name.split('/')[0]
            if('input' in iname):
                return
            shift_list[iname] = Qmin
            if(not is_shift_layer(LM[iname])):
                update_previous_layer_shift(LM[iname], Q)
    for layer in L:
        if(type(layer.input) == list):
            iname = layer.input[0].name.split('/')[0]
            Qmin = shift_list[iname]
            for inp in layer.input:
                iname = inp.name.split('/')[0]
                if(shift_list[iname] < Qmin):
                    Qmin = shift_list[iname]
                if(shift_list[iname] != Qmin):
                    bFlag = True
            for inp in layer.input:
                iname = inp.name.split('/')[0]
                shift_list[iname] = Qmin
                if(not is_shift_layer(LM[iname])):
                    update_previous_layer_shift(LM[iname], Qmin)
            print('set shift', Qmin, 'for the input of', layer.name, ':', [inp.name.split('/')[0] for inp in layer.input])
            if(not is_shift_layer(layer) or Qmin < shift_list[layer.name]): # update current layer's shift only when we cannot change the shift
                shift_list[layer.name] = Qmin
    print("shift list", shift_list)
    return shift_list

def generate_model(model, x_test, name='weights.h', format='hwc', kld=True):
    shift_list = layers_output_ranges(model, x_test, kld)
    generate_weights(model, name=name, format=format, shift_list=shift_list)
    if(type(model.layers[0]) != InputLayer):
        L = [model.input] + model.layers
    else:
        L = model.layers
    with open(name,'a') as fp:
        fp.write('\n/* output enconding for each layer */\n')
        for layer in L:
            if(model.input == layer and type(model.layers[0]) != InputLayer):
                iname = layer.name.split(':')[0]
            else:
                iname = layer.name
            fp.write('#define %s_OUTPUT_SHIFT %s\n'%(iname.upper(), shift_list[iname]))
        fp.write('\n/* bias shift and output shift for each layer */\n')
        for layer in model.layers:
            if(is_shift_layer(layer)):
                iname = layer.name.upper()
                if(len(layer.weights) == 2 and
                   'kernel' in layer.weights[0].name and
                   'bias' in layer.weights[1].name):
                    kname = layer.weights[0].name.upper().replace('/', '_').replace(':', '_')
                    bname = layer.weights[1].name.upper().replace('/', '_').replace(':', '_')
                    inp = layer.input.name.replace(':','/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT+{2}_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                            iname, inp, kname))
                    fp.write('#define {0}_BIAS_LSHIFT   ({1}_OUTPUT_SHIFT+{2}_SHIFT-{3}_SHIFT)\n'.format(
                            iname, inp, kname, bname))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                    fp.write('#if {0}_BIAS_LSHIFT < 0\n#error {0}_BIAS_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                # add, sub
                elif ('add' in layer.name or
                    'subtract' in layer.name):
                    # only consider the first, they have been set to same in out_put_range()
                    inp = layer.input[0].name.replace(':','/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                            iname, inp))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                # mult is different, Q3.4 * Q3.4 = Q6.8. if mult out is Q4.3, then shift (Q.4+q.4)-Q.3=5. Am I right?
                elif ('multiply' in layer.name ):
                    inp = layer.input[0].name.replace(':','/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT*2-{0}_OUTPUT_SHIFT)\n'.format(
                            iname, inp))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))


        fp.write('\n/* weights for each layer */\n')
        LI = {}
        ID = 0
        def is_skipable_layer(layer):
            # FIXME: add more that could be skiped
            if('lambda' in layer.name or
               'dropout' in layer.name or
               'batch_normalization' in layer.name or
                ('flatten' in layer.name and 'chw' not in format)): # flatten layer can be skipped in HWC but have to present in CHW
                return True
            return False
        for id,layer in enumerate(L):
            if(is_skipable_layer(layer)):
                inp = layer.input.name.replace(':','/').split('/')[0]
                LI[layer.name] = (LI[inp][0], layer)
            else:
                if(model.input == layer and type(model.layers[0]) != InputLayer):
                    LI[layer.name.split(':')[0]] = (ID, layer)
                else:
                    LI[layer.name] = (ID, layer)
                ID += 1

            if ('input' in layer.name or not layer.weights):
                continue
            for var in layer.weights:
                var_name = str(var.name).replace('/', '_').replace(':', '_')
                if("kernel" in var_name):
                    fp.write('static const int8_t %s_weights[] = %s;\n'%(layer.name, var_name.upper()))
                    fp.write('static const nnom_weight_t %s_w = { (const void*)%s_weights, %s_OUTPUT_RSHIFT};\n'%(layer.name,layer.name, layer.name.upper()))
                elif("bias" in var_name):
                    fp.write('static const int8_t %s_bias[] = %s;\n'%(layer.name, var_name.upper()))
                    fp.write('static const nnom_bias_t %s_b = { (const void*)%s_bias, %s_BIAS_LSHIFT};\n'%(layer.name,layer.name, layer.name.upper()))
        fp.write('\n/* nnom model */\n')
        # FIXME: now only support one input and one output
        sz = 1
        for d in model.input.shape[1:]:
            sz = sz*d
        fp.write('static int8_t nnom_input_data[%d];\n'%(sz))
        sz = 1
        for d in model.output.shape[1:]:
            sz = sz*d
        fp.write('static int8_t nnom_output_data[%d];\n'%(sz))
        fp.write('static nnom_model_t* nnom_model_create(void)\n{\n')
        fp.write('\tstatic nnom_model_t model;\n')
        if(ID>32):
            fp.write('\tnnom_layer_t ** layer = malloc(sizeof(nnom_layer_t *)*%d);\n'%(ID+1))
            fp.write('\tif(NULL == layer) return NULL;\n')
        else:
            fp.write('\tnnom_layer_t* layer[%d];\n'%(ID+1))
        fp.write('\n\tnew_model(&model);\n\n')
        for layer in L:
            if(is_skipable_layer(layer)):
                continue
            #FIXME: need a better solution to seperate the input 'tensor' from other layers
            if (model.input == layer and type(model.layers[0]) != InputLayer):
                id,_ = LI[layer.name.split(':')[0]]
            else:
                id,_ = LI[layer.name]

            if('input' in layer.name):
                try:
                    inshape = layer.input_shape[1:]
                except:
                    inshape = layer.shape[1:]
                if (len(inshape) == 1):  # 1-D input
                    fp.write('\tlayer[%d] = Input(shape(%d,1,1), nnom_input_data);\n' % (id, inshape[0]))
                elif (len(inshape) == 2):  # 1-D input
                    fp.write('\tlayer[%d] = Input(shape(1,%d,%d), nnom_input_data);\n' % (id, inshape[0], inshape[1]))
                else:
                    fp.write('\tlayer[%d] = Input(shape%s, nnom_input_data);\n' % (id, inshape))

            # convlutional
            elif('conv1d' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('depthwise' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(DW_Conv2D({1}, kernel(1,{2}), stride(1,{3}), PADDING_{4}, &{5}_w, &{5}_b), layer[{6}]);\n'.format(
                        id, 1, cfg['kernel_size'][0], cfg['strides'][0], cfg['padding'].upper(),
                        layer.name, LI[inp][0]))
                else:
                    fp.write('\tlayer[{0}] = model.hook(Conv2D({1}, kernel(1,{2}), stride(1,{3}), PADDING_{4}, &{5}_w, &{5}_b), layer[{6}]);\n'.format(
                        id, cfg['filters'], cfg['kernel_size'][0], cfg['strides'][0], cfg['padding'].upper(),
                        layer.name, LI[inp][0]))
            elif('conv2d' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if ('depthwise' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(DW_Conv2D({1}, kernel{2}, stride{3}, PADDING_{4}, &{5}_w, &{5}_b), layer[{6}]);\n'.format(
                        id, 1, cfg['kernel_size'], cfg['strides'], cfg['padding'].upper(),
                        layer.name, LI[inp][0]))
                else:
                    fp.write('\tlayer[{0}] = model.hook(Conv2D({1}, kernel{2}, stride{3}, PADDING_{4}, &{5}_w, &{5}_b), layer[{6}]);\n'.format(
                        id, cfg['filters'], cfg['kernel_size'], cfg['strides'], cfg['padding'].upper(),
                        layer.name, LI[inp][0]))
            # activations
            elif('activation' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if(cfg['activation'] == 'relu'):
                    fp.write('\tlayer[%s] = model.active(act_relu(), layer[%s]);\n'%(id, LI[inp][0]))
                if(cfg['activation'] == 'tanh'):
                    fp.write('\tlayer[%s] = model.active(act_tanh(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
                if(cfg['activation'] == 'sigmoid'):
                    fp.write('\tlayer[%s] = model.active(act_sigmoid(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
                elif(cfg['activation'] == 'softmax'):
                    fp.write('\tlayer[%s] = model.hook(Softmax(), layer[%s]);\n'%(id, LI[inp][0]))
            elif('re_lu' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                fp.write('\tlayer[%s] = model.active(act_relu(), layer[%s]);\n'%(id, LI[inp][0]))
            # pooling
            elif('max_pooling' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if ('global' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(GlobalMaxPool(),  layer[%s]);\n' % (id, LI[inp][0]))
                elif('2d' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(MaxPool(kernel%s, stride%s, PADDING_%s), layer[%d]);\n'%(
                        id, cfg['pool_size'], cfg['strides'], cfg['padding'].upper(), LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(MaxPool(kernel(1,{1}), stride(1,{2}), PADDING_{3}), layer[{4}]);\n'.format(
                        id, cfg['pool_size'][0], cfg['strides'][0], cfg['padding'].upper(), LI[inp][0]))
            elif('average_pooling' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if ('global' in layer.name):
                    # a global avg pool before softmax can be replace by sumpool in MCU (recommend)
                    if(layer == model.layers[-2] and 'Softmax' in model.layers[-1].output.name):
                        print(layer.name, 'has been replaced by GlobalSumPool()')
                        fp.write('\tlayer[%s] = model.hook(GlobalSumPool(),  layer[%s]);\n' % (id, LI[inp][0]))
                    else:
                        fp.write('\tlayer[%s] = model.hook(GlobalAvgPool(),  layer[%s]);\n' % (id, LI[inp][0]))
                elif('2d' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(AvgPool(kernel%s, stride%s, PADDING_%s), layer[%d]);\n'%(
                        id, cfg['pool_size'], cfg['strides'], cfg['padding'].upper(), LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(AvgPool(kernel(1,{1}), stride(1,{2}), PADDING_{3}), layer[{4}]);\n'.format(
                        id, cfg['pool_size'][0], cfg['strides'][0], cfg['padding'].upper(), LI[inp][0]))
            elif ('up_sampling' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('2d' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(UpSample(kernel%s), layer[%d]);\n'%(id, cfg['size'],  LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(UpSample(kernel(1,{1})), layer[{2}]);\n'.format(
                        id,  cfg['size'][0], LI[inp][0]))
            # zero padding
            elif ('zero_padding' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('2d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(ZeroPadding(border({1},{2},{3},{4})), layer[{5}]);\n'.format(
                        id,  cfg['padding'][0][0], cfg['padding'][0][1], cfg['padding'][1][0],cfg['padding'][1][1], LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(ZeroPadding(border(0,0,{1},{2})), layer[{3}]);\n'.format(
                        id,  cfg['padding'][0], cfg['padding'][1], LI[inp][0]))
            # Cropping
            elif ('cropping' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('2d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(Cropping(border({1},{2},{3},{4})), layer[{5}]);\n'.format(
                        id,  cfg['cropping'][0][0], cfg['cropping'][0][1], cfg['cropping'][1][0],cfg['cropping'][1][1], LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(Cropping(border(0,0,{1},{2})), layer[{3}]);\n'.format(
                        id,  cfg['cropping'][0], cfg['cropping'][1], LI[inp][0]))

            # others
            elif('flatten' in layer.name): # flatten is needed in CHW backend but not needed in HWC
                inp = layer.input.name.replace(':', '/').split('/')[0]
                fp.write('\tlayer[%s] = model.hook(Flatten(), layer[%s]);\n'%(id, LI[inp][0]))
            elif('concatenate' in layer.name):
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                cfg = layer.get_config()
                fp.write('\tlayer[%s] = model.mergex(Concat(%s), %s%s);\n'%(
                    id, cfg['axis'], len(inps), inX))
            elif('add' in layer.name):
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(Add(%s_OUTPUT_RSHIFT), %s%s);\n'%(
                    id, layer.name.upper(), len(inps), inX))
            elif('subtract' in layer.name):
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(Sub(%s_OUTPUT_RSHIFT), %s%s);\n'%(
                    id, layer.name.upper(), len(inps), inX))
            elif('multiply' in layer.name):
                warnings.warn("Warning mutiply is under testing")
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(Mult(%s_OUTPUT_RSHIFT), %s%s);\n'%(
                    id, layer.name.upper(), len(inps), inX))
            elif('dense' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                fp.write('\tlayer[{0}] = model.hook(Dense({1}, &{2}_w, &{2}_b), layer[{3}]);\n'.format(
                    id, cfg['units'], layer.name, LI[inp][0]))
            elif('softmax' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                fp.write('\tlayer[%s] = model.hook(Softmax(), layer[%s]);\n'%(id, LI[inp][0]))
            else:
                raise Exception('unsupported layer', layer.name, layer)
			
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
			
        # FIXME, test later.
        if('softmax' in layer.name
           or ('activation' in layer.name and layer.get_config()['activation'] == 'softmax')):
            fp.write('\tlayer[%s] = model.hook(Output(shape(%s,1,1), nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1], id))
        elif len(layer.output.shape) == 4:
            fp.write('\tlayer[%s] = model.hook(Output(shape%s, nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1:], id))
        elif len(layer.output.shape) == 3:
            fp.write('\tlayer[%s] = model.hook(Output(shape(1,%s,%s), nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1], layer.output.shape[2], id))
        elif len(layer.output.shape) == 2:
            fp.write('\tlayer[%s] = model.hook(Output(shape(%s,1,1), nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1], id))
        else:
            raise Exception('unsupported output shape of the last layer', layer.name, layer)
        fp.write('\tmodel_compile(&model, layer[0], layer[%s]);\n'%(id+1))
        if(ID>32):
            fp.write('\tfree(layer);\n')
        fp.write('\treturn &model;\n}\n')
    with open('.shift_list','w') as fp:
        fp.write(str(shift_list))

def evaluate_model(model, x_test, y_test, running_time=False, to_file='evaluation.txt'):
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', scores[0])
    print('Top 1:', scores[1])

    if(len(y_test.shape)>1):
        predictions = model.predict(x_test)
        output = tf.keras.metrics.top_k_categorical_accuracy(y_test, predictions, k=2)
        with tf.Session() as sess:
            result = sess.run(output)
        print("Top 2:",result)

        predictions = model.predict(x_test)
        matrix = metrics.confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
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
            f.write("Top 2:"+ str(result)+ "\n")
            #f.write(str(matrix))
            for row in matrix:
                row.tofile(f, sep=',')
                f.write("\n")

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
    return scores


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

