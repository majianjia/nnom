'''
    Copyright (c) 2018-2020
    Jianjia Ma
    majianjia@live.com
    SPDX-License-Identifier: Apache-2.0
    Change Logs:
    Date           Author       Notes
    2020-05-22     Jianjia Ma   The first version
'''
from tensorflow.keras.layers import *
import numpy as np

def convert_tensor_name(t):
    return 'tensor_'+t.name.replace('/', '_').replace(':', '_')

def to_cstyle(data, integer=True):
    #Convert an array to C style basket, not to be used for very large array. size > options['threshold'] will lead to ...
    if(integer):
        data = np.array(data, dtype=np.int).flatten()
    else:
        data = np.array(data).flatten()
    s = np.array2string(data, separator=',')
    s = s.replace("\n","").replace("\r","").replace(' ','')
    s = s.replace(',', ', ')
    s = s.replace('(', '[').replace(')', ']')
    return s.replace('[', '{').replace(']', '}')

def tensor_shape(tensor, is_io_tensor=False):
    # inconsistance of TF1 and TF2
    # get tensor shape without None or ?
    try:
        shape = tensor.shape.as_list() # tf1
    except:
        shape = tensor.get_shape().as_list() # tf2
    if(shape[0] == None or is_io_tensor):
        shape = shape[1:]
    else:
        shape = shape
    # for rnn input with timestamp = None, need a better implementation
    for i in range(len(shape)):
        shape[i] = shape[i] if shape[i] is not None else 1
    return shape

def gen_base_config(layer):
    config = '{.name = "%s"}' % (layer.name)
    return config

def gen_values(var_name, var, size='', dtype='const int8_t'):
    s = '<dtype> <var_name>[<size>] = <var>;\n'
    s = s.replace('<var_name>', var_name).replace('<var>', var).replace('<size>', size).replace('<dtype>', dtype)
    return s

# generate tensor by the tensor config
def gen_tensor(tensor, dec_bits, tensor_value='NULL', per_axis=False, is_io_tensor=False):
    config = '''
const nnom_shape_data_t <tensor_name>_dim[] = <dim>;
const nnom_qformat_param_t <tensor_name>_dec[] = <q_dec>;
const nnom_qformat_param_t <tensor_name>_offset[] = <q_offset>;
const nnom_tensor_t <tensor_name> = {
    .p_data = (void*)<value>,
    .dim = (nnom_shape_data_t*)<tensor_name>_dim,
    .q_dec = (nnom_qformat_param_t*)<tensor_name>_dec,
    .q_offset = (nnom_qformat_param_t*)<tensor_name>_offset,
    .qtype = <qtype>,
    .num_dim = <num_dim>,
    .bitwidth = <bitwidth>
};
'''
    # inconsistance of TF1 and TF2
    shape = tensor_shape(tensor, is_io_tensor)
    config = config.replace('<tensor_name>', convert_tensor_name(tensor))#.name.replace('/','_').split(':')[0]) #conv2d/kernel:0
    config = config.replace('<bitwidth>', '8')
    config = config.replace('<value>', tensor_value)
    config = config.replace('<dim>', to_cstyle(shape))
    config = config.replace('<num_dim>', str(len(shape)))
    if(type(dec_bits) == str):
        config = config.replace('<q_dec>', dec_bits)
        config = config.replace('<q_offset>', to_cstyle([0]))
    else:
        config = config.replace('<q_dec>', to_cstyle(dec_bits))
        config = config.replace('<q_offset>', to_cstyle([0]))
    if(per_axis):
        config = config.replace('<qtype>', 'NNOM_QTYPE_PER_AXIS')
    else:
        config = config.replace('<qtype>', 'NNOM_QTYPE_PER_TENSOR')
    return config

# create tensor by directly setting up the value
def gen_create_tensor(tensor_name, shape, dec_bits, tensor_value='NULL', per_axis=False):
    config = '''
const nnom_shape_data_t <tensor_name>_dim[] = <dim>;
const nnom_qformat_param_t <tensor_name>_dec[] = <q_dec>;
const nnom_qformat_param_t <tensor_name>_offset[] = <q_offset>;
const nnom_tensor_t <tensor_name> = {
    .p_data = (void*)<value>,
    .dim = (nnom_shape_data_t*)<tensor_name>_dim,
    .q_dec = (nnom_qformat_param_t*)<tensor_name>_dec,
    .q_offset = (nnom_qformat_param_t*)<tensor_name>_offset,
    .qtype = <qtype>,
    .num_dim = <num_dim>,
    .bitwidth = <bitwidth>
};
'''
    config = config.replace('<tensor_name>', tensor_name)
    config = config.replace('<bitwidth>', '8')
    config = config.replace('<value>', tensor_value)
    config = config.replace('<dim>', to_cstyle(shape))
    config = config.replace('<num_dim>', str(len(shape)))
    if(type(dec_bits) == str):
        config = config.replace('<q_dec>', dec_bits)
        config = config.replace('<q_offset>', to_cstyle([0]))
    else:
        config = config.replace('<q_dec>', to_cstyle(dec_bits))
        config = config.replace('<q_offset>', to_cstyle([0]))
    if(per_axis):
        config = config.replace('<qtype>', 'NNOM_QTYPE_PER_AXIS')
    else:
        config = config.replace('<qtype>', 'NNOM_QTYPE_PER_TENSOR')
    return config

def gen_conv2d_config(layer, output_shifts, bias_shifts):
    c = '''
const nnom_qformat_param_t <layer_name>_output_shift[] = <output_shift_values>;
const nnom_qformat_param_t <layer_name>_bias_shift[] = <bias_shift_values>;
const nnom_conv2d_config_t <layer_name>_config = {
    .super = <base_config>,
    .qtype = <qtype>,
    .weight = (nnom_tensor_t*)&<weight>,
    .bias = (nnom_tensor_t*)&<bias>,
    .output_shift = (nnom_qformat_param_t *)&<layer_name>_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&<layer_name>_bias_shift, 
    .filter_size = <filter_size>,
    .kernel_size = <kernel_size>,
    .stride_size = <stride_size>,
    .padding_size = <padding_size>,
    .dilation_size = <dilation_size>,
    .padding_type = <padding_type>
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<qtype>', "NNOM_QTYPE_PER_TENSOR")
    c = c.replace('<weight>',convert_tensor_name(layer.weights[0]))
    c = c.replace('<bias>',convert_tensor_name(layer.weights[1]))
    c = c.replace('<output_shift_values>', output_shifts)
    c = c.replace('<bias_shift_values>', bias_shifts)
    c = c.replace('<filter_size>', str(layer.filters) if layer.filters is not None else str(layer.depth_multiplier))  # output channel
    c = c.replace('<kernel_size>', to_cstyle(layer.kernel_size))
    c = c.replace('<stride_size>', to_cstyle(layer.strides))
    c = c.replace('<padding_size>', '{0, 0}') # not using it with keras, defined by padding type instead
    c = c.replace('<dilation_size>', to_cstyle(layer.dilation_rate))
    c = c.replace('<padding_type>', 'PADDING_'+layer.padding.upper())
    return c

def gen_conv2d_trans_config(layer, output_shifts, bias_shifts):
    c = '''
const nnom_qformat_param_t <layer_name>_output_shift[] = <output_shift_values>;
const nnom_qformat_param_t <layer_name>_bias_shift[] = <bias_shift_values>;
const nnom_conv2d_trans_config_t <layer_name>_config = {
    .super = <base_config>,
    .qtype = <qtype>,
    .weight = (nnom_tensor_t*)&<weight>,
    .bias = (nnom_tensor_t*)&<bias>,
    .output_shift = (nnom_qformat_param_t *)&<layer_name>_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&<layer_name>_bias_shift, 
    .filter_size = <filter_size>,
    .kernel_size = <kernel_size>,
    .stride_size = <stride_size>,
    .padding_size = <padding_size>,
    .dilation_size = <dilation_size>,
    .padding_type = <padding_type>
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<qtype>', "NNOM_QTYPE_PER_TENSOR")
    c = c.replace('<weight>',convert_tensor_name(layer.weights[0]))
    c = c.replace('<bias>',convert_tensor_name(layer.weights[1]))
    c = c.replace('<output_shift_values>', output_shifts)
    c = c.replace('<bias_shift_values>', bias_shifts)
    c = c.replace('<filter_size>', str(layer.filters)) # output channel
    c = c.replace('<kernel_size>', to_cstyle(layer.kernel_size))
    c = c.replace('<stride_size>', to_cstyle(layer.strides))
    c = c.replace('<padding_size>', '{0, 0}') # not using it with keras, defined by padding type instead
    c = c.replace('<dilation_size>', to_cstyle(layer.dilation_rate))
    c = c.replace('<padding_type>', 'PADDING_'+layer.padding.upper())
    return c

def gen_dense_config(layer, output_shifts, bias_shift):
    c = '''
const nnom_qformat_param_t <layer_name>_output_shift[] = <output_shift_values>;
const nnom_qformat_param_t <layer_name>_bias_shift[] = <bias_shift_values>;
const nnom_dense_config_t <layer_name>_config = {
    .super = <base_config>,
    .qtype = <qtype>,
    .weight = (nnom_tensor_t*)&<weight>,
    .bias = (nnom_tensor_t*)&<bias>,
    .output_shift = (nnom_qformat_param_t *)&<layer_name>_output_shift,
    .bias_shift = (nnom_qformat_param_t *)&<layer_name>_bias_shift
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<qtype>', "NNOM_QTYPE_PER_TENSOR")
    c = c.replace('<weight>', convert_tensor_name(layer.weights[0]))
    c = c.replace('<bias>', convert_tensor_name(layer.weights[1]))
    c = c.replace('<output_shift_values>', output_shifts)
    c = c.replace('<bias_shift_values>', bias_shift)
    return c

def gen_io_config(layer, tensor_name):
    c = '''
const nnom_io_config_t <layer_name>_config = {
    .super = <base_config>,
    .tensor = (nnom_tensor_t*)&<tensor>
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<tensor>', tensor_name)
    return c

def gen_output_config(previous_layer, dec_bits, output_num, value_name='nnom_output_data'): #cheat at the moments
    c = '''
const nnom_shape_data_t <tensor_name>_dim[] = <dim>;
const nnom_qformat_param_t <tensor_name>_dec[] = <q_dec>;
const nnom_qformat_param_t <tensor_name>_offset[] = <q_offset>;
const nnom_tensor_t <tensor_name> = {
    .p_data = (void*)<value>,
    .dim = (nnom_shape_data_t*)<tensor_name>_dim,
    .q_dec = (nnom_qformat_param_t*)<tensor_name>_dec,
    .q_offset = (nnom_qformat_param_t*)<tensor_name>_offset,
    .qtype = <qtype>,
    .num_dim = <num_dim>,
    .bitwidth = 8
};

const nnom_io_config_t <layer_name>_config = {
    .super = <base_config>,
    .tensor = (nnom_tensor_t*)&<tensor_name>
};
'''
    shape = tensor_shape(previous_layer.output, is_io_tensor=True)

    c = c.replace('<tensor_name>', 'tensor_output'+str(output_num))
    c = c.replace('<layer_name>', 'output'+str(output_num))
    c = c.replace('<base_config>', '{.name = "output'+str(output_num)+'"}') # cheating at the moment.
    c = c.replace('<value>', value_name)
    c = c.replace('<qtype>', 'NNOM_QTYPE_PER_TENSOR')
    c = c.replace('<num_dim>', str(len(shape)))
    c = c.replace('<dim>', to_cstyle(shape))
    c = c.replace('<q_dec>', '{'+dec_bits+'}')
    c = c.replace('<q_offset>', to_cstyle([0]))
    return c


def gen_pooling_config(layer, output_shifts='0'):
    c = '''
const nnom_pool_config_t <layer_name>_config = {
    .super = <base_config>,
    .padding_type = <padding_type>,
    .output_shift = <output_shift>,
    .kernel_size = <kernel_size>,
    .stride_size = <stride_size>,
    .num_dim = <num_dim>
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<padding_type>', 'PADDING_'+layer.padding.upper())
    c = c.replace('<kernel_size>', to_cstyle(layer.pool_size))
    c = c.replace('<stride_size>', to_cstyle(layer.strides))
    c = c.replace('<num_dim>', str(len(layer.pool_size)))
    c = c.replace('<output_shift>', output_shifts) # not used at the moment
    return c

def gen_gl_pooling_config(layer, output_shifts='0'):
    c = '''
const nnom_global_pool_config_t <layer_name>_config = {
    .super = <base_config>,
    .output_shift = <output_shift>,
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<output_shift>', output_shifts)
    return c



def gen_matrix_config(layer, output_shift_name='0'):
    c = '''
const nnom_matrix_config_t <layer_name>_config = {
    .super = <base_config>,
    .output_shift = <output_shift> 
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<output_shift>',  output_shift_name) # not used at the moment
    return c

def gen_zero_padding_config(layer):
    c = '''
const nnom_zero_padding_config_t <layer_name>_config = {
    .super = <base_config>,
    .pad = <padding> 
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    try:
        c = c.replace('<padding>', to_cstyle(sum(layer.padding, ())))
    except:
        pad = ((0, 0), layer.padding)
        c = c.replace('<padding>', to_cstyle(sum(pad, ())))
    return c

def gen_cropping_config(layer):
    c = '''
const nnom_cropping_config_t <layer_name>_config = {
    .super = <base_config>,
    .pad = <padding>
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    try:
        c = c.replace('<padding>', to_cstyle(sum(layer.cropping, ()))) #((top_crop, bottom_crop), (left_crop, right_crop))
    except:
        pad = ((0, 0), layer.cropping)
        c = c.replace('<padding>', to_cstyle(sum(pad, ())))
    return c

def gen_upsampling_config(layer):
    c = '''
const nnom_upsample_config_t <layer_name>_config = {
    .super = <base_config>,
    .kernel = <kernel> 
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<kernel>', to_cstyle(layer.size))
    return c

def gen_softmax_config(layer):
    c = '''
const nnom_softmax_config_t <layer_name>_config = {
    .super = <base_config>
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    return c

def gen_flatten_config(layer):
    c = '''
const nnom_flatten_config_t <layer_name>_config = {
    .super = <base_config>
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    return c

def gen_concat_config(layer):
    c = '''
const nnom_concat_config_t <layer_name>_config = {
    .super = <base_config>,
    .axis = <axis>
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<axis>', str(layer.axis))
    return c

def gen_lambda_config(layer, run_func_name='NULL', build_func_name='NULL', free_func_name='NULL', parameters_name='NULL'):
    c = '''
const nnom_lambda_config_t <layer_name>_config = {
    .super = <base_config>,
    .run_func_name = <run_func_name>,
    .build_func_name = <build_func_name>,
    .free_func_name = <free_func_name>,
    .parameters = <parameters_name>
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<run_func_name>', run_func_name)
    c = c.replace('<build_func_name>', build_func_name)
    c = c.replace('<free_func_name>', free_func_name)
    c = c.replace('<parameters_name>', parameters_name)
    return c

def gen_rnn_config(layer):
    c = '''
const nnom_rnn_config_t <layer_name>_config = {
    .super = <base_config>,
    .stateful = <stateful>,
    .return_sequence = <return_sequence>,
    .go_backwards = <go_backwards>
};
'''
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<stateful>', 'true' if layer.stateful else 'false')
    c = c.replace('<go_backwards>', 'true' if layer.go_backwards else 'false')
    c = c.replace('<return_sequence>', 'true' if layer.return_sequences else 'false')
    return c

def gen_simple_cell_config(layer, q_list):
    c = '''
const nnom_simple_cell_config_t <layer_name>_simple_cell_config = {
    .super = <base_config>,
    .weights = (nnom_tensor_t*)&<weights>,
    .recurrent_weights = (nnom_tensor_t*)&<recurrent_weights>,
    .bias = (nnom_tensor_t*)&<bias>,
    .q_dec_iw = <q_dec_iw>,
    .q_dec_hw = <q_dec_hw>,
    .q_dec_h = <q_dec_h>,
    .act_type = <act_type>,
    .units = <units>
};
'''
    try:
        cell_cfg = layer.get_config()['cell']['config']
    except:
        cell_cfg = layer.get_config()
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<weights>', convert_tensor_name(layer.weights[0]))
    c = c.replace('<recurrent_weights>', convert_tensor_name(layer.weights[1]))
    c = c.replace('<bias>', convert_tensor_name(layer.weights[2]))
    c = c.replace('<q_dec_iw>', str(q_list[1])) # the qfmt of input x weight
    c = c.replace('<q_dec_hw>', str(q_list[2])) # q of hidden x recurrent weight
    c = c.replace('<q_dec_h>', str(q_list[0])) # output, if act != relu, should be 7 (consider delete it.)
    c = c.replace('<act_type>', 'ACT_' + cell_cfg['activation'].upper())
    c = c.replace('<units>', str(cell_cfg['units']))
    return c

def gen_lstm_cell_config(layer, q_list):
    c = '''
const nnom_lstm_cell_config_t <layer_name>_lstm_cell_config = {
    .super = <base_config>,
    .weights = (nnom_tensor_t*)&<weights>,
    .recurrent_weights = (nnom_tensor_t*)&<recurrent_weights>,
    .bias = (nnom_tensor_t*)&<bias>,
    .q_dec_z = <q_dec_z>,
    .q_dec_h = <q_dec_h>,
    .q_dec_c = <q_dec_c>,
    .units = <units>
};
'''
    try:
        cell_cfg = layer.get_config()['cell']['config']
    except:
        cell_cfg = layer.get_config()
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<weights>', convert_tensor_name(layer.weights[0]))
    c = c.replace('<recurrent_weights>', convert_tensor_name(layer.weights[1]))
    c = c.replace('<bias>', convert_tensor_name(layer.weights[2]))
    c = c.replace('<q_dec_h>', str(q_list[0])) # output and memory state, (should be q0.7. consider delete it)
    c = c.replace('<q_dec_c>', str(q_list[1])) # cell state
    c = c.replace('<q_dec_z>', str(q_list[2])) # input*weight + hidden*weight + bias
    c = c.replace('<units>', str(cell_cfg['units']))
    return c



def gen_gru_cell_config(layer, q_list):
    c = '''
const nnom_gru_cell_config_t <layer_name>_gru_cell_config = {
    .super = <base_config>,
    .weights = (nnom_tensor_t*)&<weights>,
    .recurrent_weights = (nnom_tensor_t*)&<recurrent_weights>,
    .bias = (nnom_tensor_t*)&<bias>,
    .q_dec_z = <q_dec_z>,
    .q_dec_h = <q_dec_h>,
    .units = <units>
};
'''
    try:
        cell_cfg = layer.get_config()['cell']['config']
    except:
        cell_cfg = layer.get_config()
    c = c.replace('<layer_name>', layer.name)
    c = c.replace('<base_config>', gen_base_config(layer))
    c = c.replace('<weights>', convert_tensor_name(layer.weights[0]))
    c = c.replace('<recurrent_weights>', convert_tensor_name(layer.weights[1]))
    c = c.replace('<bias>', convert_tensor_name(layer.weights[2]))
    c = c.replace('<q_dec_h>', str(q_list[0])) #
    c = c.replace('<q_dec_z>', str(q_list[1])) #
    c = c.replace('<units>', str(cell_cfg['units']))
    return c


if __name__ == "__main__":
    # test only
    from tensorflow.keras.models import load_model
    model = load_model("../model.h5")
    print(gen_tensor(model.layers[1].weights[0], dec_bits=(1, 2, 3, 4, 5)))
    print(gen_tensor(model.layers[1].weights[1], dec_bits=(1, 2, 3, 4, 5)))
    print(gen_conv2d_config(model.layers[1], (1,2,3), 3))

    with open("test.h", 'w') as fp:
        # fp.write(gen_tensor(model.layers[1].weights[0], dec_bits=(1, 2, 3, 4, 5)))
        # fp.write(gen_tensor(model.layers[1].weights[1], dec_bits=(1, 2, 3, 4, 5)))
        # fp.write(gen_conv2d_config(model.layers[1], (1,2,3,)))

        fp.write('#include "nnom.h"\n')

        # test all
        for layer in model.layers:
            if(type(layer) in [Conv2D, Conv1D]):
                for w in layer.weights:
                    fp.write(gen_tensor(w, [3]))
                fp.write(gen_conv2d_config(layer, {0}, 2))
            elif(type(layer) in [Dense]):
                for w in layer.weights:
                    fp.write(gen_tensor(w, [3]))
                fp.write(gen_dense_config(layer, 2, 2))
            elif(type(layer) in [Input]):
                fp.write(gen_io_config(layer, [9,1,1]))
            elif(type(layer) in [MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D, GlobalAveragePooling2D]):
                fp.write(gen_pooling_config(layer))
            elif(type(layer) in [Multiply, Add, Subtract]):
                fp.write(gen_matrix_config(layer))
            elif(type(layer) in [ZeroPadding2D, ZeroPadding1D]):
                fp.write(gen_zero_padding_config(layer))
            elif(type(layer) in [Cropping2D, Cropping1D]):
                fp.write(gen_cropping_config(layer))
            elif(type(layer) in [Softmax]):
                fp.write(gen_softmax_config(layer))
            elif(type(layer) in [Flatten]):
                fp.write(gen_flatten_config(layer))
            elif(type(layer) in [Concatenate]):
                fp.write(gen_concat_config(layer))
            elif(type(layer) in [Lambda]):
                fp.write(gen_lambda_config(layer))
            elif(type(layer) in [UpSampling2D, UpSampling1D]):
                fp.write(gen_upsampling_config(layer))


