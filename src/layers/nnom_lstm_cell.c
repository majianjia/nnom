/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-08-24     Jianjia Ma   The first version
 */

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_layers.h"
#include "layers/nnom_lstm_cell.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

// LSTM RNN
// unit = output shape
// type of activation
nnom_rnn_cell_t *lstm_cell_s(const nnom_lstm_cell_config_t* config)
{
	nnom_lstm_cell_t *cell;
	cell = nnom_mem(sizeof(nnom_lstm_cell_t));
	if (cell == NULL)
		return NULL;
	// set methods
	cell->super.run = lstm_cell_run;
	cell->super.build = lstm_cell_build;
	cell->super.free = lstm_cell_free;
	cell->super.config = (void*) config;
	cell->super.units = config->units;
    cell->super.type = NNOM_LSTM_CELL;

	// set parameters 
	cell->bias = config->bias;
	cell->weights = config->weights;
	cell->recurrent_weights = config->recurrent_weights;
	cell->act_type = config->act_type; 
	// // q format for intermediate products
	// cell->q_dec_iw = config->q_dec_iw;
	// cell->q_dec_hw = config->q_dec_hw;
	// cell->q_dec_h = config->q_dec_h;
	
	return (nnom_rnn_cell_t *)cell;
}

nnom_status_t lstm_cell_free(nnom_rnn_cell_t* cell)
{
	return NN_SUCCESS;
}

// the state buffer and computational buffer shape of the cell
nnom_status_t lstm_cell_build(nnom_rnn_cell_t* cell)
{
	nnom_layer_t *layer = cell->layer; 
	nnom_lstm_cell_t *c = (nnom_lstm_cell_t *)cell;
	nnom_lstm_cell_config_t *config = (nnom_lstm_cell_config_t *)cell->config;
	int q_hw_iw;
	
	// activation, check if activation is supported 
	if(config->act_type != ACT_SIGMOID && config->act_type != ACT_TANH)
		return NN_ARGUMENT_ERROR;

	// calculate output shift for the 2 calculation. 
	// hw = the product of hidden x weight, iw = the product of input x weight
	// due to the addition of them, they must have same q format.
	q_hw_iw = MIN(c->q_dec_hw, c->q_dec_iw);  

	// for the 2 dot in cell: output shift = input_dec + weight_dec - output_dec
	c->oshift_hw = c->q_dec_h + c->recurrent_weights->q_dec[0] - q_hw_iw;
	c->oshift_iw = layer->in->tensor->q_dec[0] + c->weights->q_dec[0] - q_hw_iw;

	// bias shift =  bias_dec - out_dec
	c->bias_shift = layer->in->tensor->q_dec[0] + c->weights->q_dec[0] - c->bias->q_dec[0];

	// state size = one timestamp output size. 
	cell->state_size = cell->units * 2;

	// // comp buffer size: not required
	cell->comp_buf_size = cell->units * 12; 

	// // finally, calculate the MAC for info
	cell->macc = tensor_size(layer->in->tensor) * cell->units *4 	  // input: (feature * timestamp) * state * 4 gates
				+ cell->state_size * tensor_size(layer->out->tensor) *4;  // recurrent, state * (timestamp * output_unit) * 4 gate

	return NN_SUCCESS;
}


// keras implementation as below. 
/*
  def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]  # previous memory state
    c_tm1 = cell_states[1]  # previous carry state

    z = K.dot(cell_inputs, kernel)          -> q_iw
    z += K.dot(h_tm1, recurrent_kernel)     -> q_hw
    z = K.bias_add(z, bias)                 

    z0, z1, z2, z3 = array_ops.split(z, 4, axis=1)

    i = nn.sigmoid(z0)
    f = nn.sigmoid(z1)
    c = f * c_tm1 + i * nn.tanh(z2)
    o = nn.sigmoid(z3)

    h = o * nn.tanh(c)
    return h, [h, c]
*/

nnom_status_t lstm_cell_run(nnom_rnn_cell_t* cell)
{
	nnom_layer_t *layer = cell->layer;
	nnom_rnn_layer_t* cl = (nnom_rnn_layer_t *) layer;
	nnom_lstm_cell_t* c = (nnom_lstm_cell_t*) cell;
    int act_int_bit = 7 - MIN(c->q_dec_hw, c->q_dec_iw);

    // state buffer
    // low |-- hidden --|-- carry --| high
    q7_t* h_tm1 = (q7_t*)cell->in_state;
    q7_t* c_tm1 = (q7_t*)cell->in_state + cell->state_size/2;
    q7_t* o_state[2];
    o_state[0] = (q7_t*)cell->out_state;
    o_state[1] = (q7_t*)cell->out_state + cell->state_size/2;

    // computing buffer
    // low |-- buf0 --|-- buf1 --|-- buf2 --|
    q7_t* z[4];
    q7_t *buf0, *buf1, *buf2;
    buf0 = (q7_t*)layer->comp->mem->blk;
    buf1 = (q7_t*)layer->comp->mem->blk + cell->units*4;
    buf2 = (q7_t*)layer->comp->mem->blk + cell->units*8;

    // z = K.dot(h_tm1, recurrent_kernel)  -> buf1
    local_dot_q7_opt(h_tm1, c->recurrent_weights->p_data, cell->units*4, cell->units*4, c->oshift_hw, buf1);

    // z1 = K.dot(cell_inputs, kernel) + bias -> buf2
    local_fully_connected_q7_opt(cell->in_data, c->weights->p_data, 
            cell->feature_size, cell->units*4, c->bias_shift, c->oshift_iw, c->bias->p_data, buf2, NULL);

    // z += z1  -> buf0
    local_add_q7(buf1, buf2, buf0, 0, cell->units*4);

    // split the data to each gate
    z[0] = buf0;
    z[1] = buf0 + cell->units;
    z[2] = buf0 + cell->units*2;
    z[3] = buf0 + cell->units*3;

    // i = nn.sigmoid(z0)
    local_sigmoid_q7(z[0], cell->units, act_int_bit);
    // f = nn.sigmoid(z1)
    local_sigmoid_q7(z[1], cell->units, act_int_bit);
    // o = nn.sigmoid(z3)
    local_sigmoid_q7(z[3], cell->units, act_int_bit);

    // c = f * c_tm1 + i * nn.tanh(z2)
    // 1. i * tanh(z2) -> buf1
    local_tanh_q7(z[2], cell->units, act_int_bit);
    local_dot_q7_opt(z[2], z[0], cell->units, cell->units, 7, buf1);

    // 2. f * c_tm1 -> o_state[0] 
    local_dot_q7_opt(z[1], c_tm1, cell->units, cell->units, 7, o_state[0]); //q0.7 * q0.7 shift 7

    // 3. c = i*tanh + f*c_tm1 -> o_state[1] ** fill the upper state (carry)
    local_add_q7(buf1, o_state[0], o_state[1], 0, cell->units);

    // h = o * nn.tanh(c) -> o_state[0] ** fill the lower state (memory, hidden)
    local_tanh_q7(o_state[1], cell->units, 0); // should be Q0.7 check later. 
    local_dot_q7_opt(z[3], o_state[1], cell->units, cell->units, 7, o_state[0]);

    // h -> output ** (copy hidden to output)
    memcpy(cell->out_data, o_state[1], cell->units);

	return NN_SUCCESS;
}


