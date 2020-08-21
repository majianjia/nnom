/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-08-21     Jianjia Ma   The first version
 */

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_layers.h"
#include "layers/nnom_simple_cell.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

// Simple RNN
// unit = output shape
// type of activation
nnom_rnn_cell_t *simple_cell_s(nnom_simple_cell_config_t* config)
{
	nnom_simple_cell_t *cell;
	cell = nnom_mem(sizeof(nnom_simple_cell_t));
	if (cell == NULL)
		return (nnom_rnn_cell_t *)cell;
	// set methods
	cell->super.run = simple_cell_run;
	cell->super.build = simple_cell_build;
	cell->super.free = simple_cell_free;
	cell->super.config = (void*) config;
	cell->super.units = config->units;

	// set parameters 
	cell->bias = config->bias;
	cell->weights = config->weight;
	cell->recurrent_weights = config->recurrent_weights;
	cell->act_type = config->act_type; 

	//cell->output_shift = w->shift;
	//cell->bias_shift = w->shift - b->shift;	// bias is quantized to have maximum shift of weights

	return (nnom_rnn_cell_t *)cell;
}


nnom_status_t simple_cell_free(nnom_rnn_cell_t* cell)
{
	// nnom_simple_cell_t *c = (nnom_simple_cell_t*)cell;
	// act_delete(c->act);
	return NN_SUCCESS;
}


// the state buffer and computational buffer shape of the cell
nnom_status_t simple_cell_build(nnom_rnn_cell_t* cell)
{
	nnom_layer_t *layer = cell->layer; 
	nnom_simple_cell_t *c = (nnom_simple_cell_t *)cell;
	nnom_simple_cell_config_t *config = (nnom_simple_cell_config_t *)cell->config;
	
	// activation, check if activation is supported 
	switch(config->act_type)
	{
		case ACT_SIGMOID: //c->act = act_sigmoid(int_bit); 
		break;
		case ACT_TANH:// c->act = act_tanh(int_bit); 
		break;
		default: 
			return NN_ARGUMENT_ERROR;
	}


	// finally, calculate the MAC for info
	cell->macc = tensor_size(layer->in->tensor) * tensor_size(layer->out->tensor) + tensor_size(layer->out->tensor)*tensor_size(layer->out->tensor);

	return NN_SUCCESS;
}

// This Simple Cell replicate the Keras's SimpleCell as blow 
/*
 def call(self, inputs, states, training=None):
    prev_output = states[0] if nest.is_sequence(states) else states

	h = K.dot(inputs, self.kernel)
	h = K.bias_add(h, self.bias)

	h2 = K.dot(prev_output, self.recurrent_kernel)
    output = h + H2
    output = self.activation(output)

    new_state = [output] if nest.is_sequence(states) else output
    return output, new_state
*/

nnom_status_t simple_cell_run(nnom_rnn_cell_t* cell)
{
	nnom_layer_t *layer = cell->layer;
	nnom_rnn_layer_t* cl 	= (nnom_rnn_layer_t *)layer;
	nnom_simple_cell_t* c = (nnom_simple_cell_t*) cell;
	int act_int_bit = 7 - c->q_dec_hw;

	// in_state x recurrent_weight -> h2 (output buf)
	local_dot_q7_opt(cell->in_state, c->recurrent_weights->p_data, c->units, cell->units, c->oshift_hw, cell->out_data);
	// (input x weight) + bias -> h (in_state buf)
	local_fully_connected_q7_opt(cell->in_data, c->weights->p_data, c->vsize, cell->units, c->bias_shift, c->oshift_iw, c->bias->p_data, cell->in_state, NULL);
	// h + h2 -> (out_state buf)
	local_add_q7(cell->in_state, cell->out_data, cell->out_state, 0, cell->units);

	// active(out_state buf)
	if(c->act_type == ACT_TANH)
		local_tanh_q7(cell->out_state, cell->units, act_int_bit);
	else
		local_sigmoid_q7(cell->out_state, cell->units, act_int_bit);

	// (out_state buf) --copy--> (output buf)
	memcpy(cell->out_data, cell->out_state, cell->units);

	return NN_SUCCESS;
}


