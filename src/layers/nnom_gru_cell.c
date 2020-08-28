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
#include "layers/nnom_gru_cell.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

nnom_rnn_cell_t *gru_cell_s(const nnom_gru_cell_config_t* config)
{
	nnom_gru_cell_t *cell;
	cell = nnom_mem(sizeof(nnom_gru_cell_t));
	if (cell == NULL)
		return NULL;
	// set methods
	cell->super.run = gru_cell_run;
	cell->super.build = gru_cell_build;
	cell->super.free = gru_cell_free;
	cell->super.config = (void*) config;
	cell->super.units = config->units;
    cell->super.type = NNOM_GRU_CELL;

	// set parameters 
	cell->bias = config->bias;
	cell->weights = config->weights;
	cell->recurrent_weights = config->recurrent_weights;

    // q format for intermediate calculation
    cell->q_dec_r = config->q_dec_r;
    cell->q_dec_h = config->q_dec_h;
    cell->q_dec_z = config->q_dec_z;

	// // q format for intermediate products
	// cell->q_dec_iw = config->q_dec_iw;
	// cell->q_dec_hw = config->q_dec_hw;
	// cell->q_dec_h = config->q_dec_h;
	
	return (nnom_rnn_cell_t *)cell;
}

nnom_status_t gru_cell_free(nnom_rnn_cell_t* cell)
{
	return NN_SUCCESS;
}

// the state buffer and computational buffer shape of the cell
nnom_status_t gru_cell_build(nnom_rnn_cell_t* cell)
{
	nnom_layer_t *layer = cell->layer; 
	nnom_gru_cell_t *c = (nnom_gru_cell_t *)cell;
	nnom_gru_cell_config_t *config = (nnom_gru_cell_config_t *)cell->config;

	// calculate output shift for the 2 calculation. 
	// hw = the product of hidden x weight, iw = the product of input x weight
	// due to the addition of them, they must have same q format.
    // that is -> c->q_dec_z; 

	// for the dots in cell: output shift = input_dec + weight_dec - output_dec
	c->oshift_hw = c->q_dec_h + c->recurrent_weights->q_dec[0] - c->q_dec_z; 
	c->oshift_iw = layer->in->tensor->q_dec[0] + c->weights->q_dec[0] - c->q_dec_z; 

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
    h_tm1 = cell_states[0]

    # inputs projected by all gate matrices at once
    matrix_x = K.dot(cell_inputs, kernel)
    matrix_x = K.bias_add(matrix_x, input_bias)

    x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=1)

    # hidden state projected by all gate matrices at once
    matrix_inner = K.dot(h_tm1, recurrent_kernel)
    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)

    recurrent_z, recurrent_r, recurrent_h = array_ops.split(matrix_inner, 3,
                                                            axis=1)
    z = nn.sigmoid(x_z + recurrent_z)
    r = nn.sigmoid(x_r + recurrent_r)
    hh = nn.tanh(x_h + r * recurrent_h)

    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    return h, [h]
*/

//
nnom_status_t gru_cell_run(nnom_rnn_cell_t* cell)
{
	nnom_layer_t *layer = cell->layer;
	nnom_rnn_layer_t* cl = (nnom_rnn_layer_t *) layer;
	nnom_gru_cell_t* c = (nnom_gru_cell_t*) cell;
    int act_int_bit = 7 - c->q_dec_z;

	
	// 			// test
	// 			//memset(cell->in_data, 32, cell->feature_size); 

    // // state buffer
    // // low |-- hidden --|-- carry --| high
    // q7_t* h_tm1 = (q7_t*)cell->in_state;
    // q7_t* c_tm1 = (q7_t*)cell->in_state + cell->state_size/2;
    // q7_t* o_state[2];
    // o_state[0] = (q7_t*)cell->out_state;
    // o_state[1] = (q7_t*)cell->out_state + cell->state_size/2;

    // // computing buffer
    // // low |-- buf0 --|-- buf1 --|-- buf2 --|
    // q7_t* z[4];
    // q7_t *buf0, *buf1, *buf2;
    // buf0 = (q7_t*)layer->comp->mem->blk;
    // buf1 = (q7_t*)layer->comp->mem->blk + cell->units*4;
    // buf2 = (q7_t*)layer->comp->mem->blk + cell->units*8;

    // // z1 = K.dot(cell_inputs, kernel) + bias -> buf2
    // local_fully_connected_q7_opt(cell->in_data, c->weights->p_data, 
    //         cell->feature_size, cell->units*4, c->bias_shift, c->oshift_iw, c->bias->p_data, buf1, NULL);

    // // z2 = K.dot(h_tm1, recurrent_kernel)  -> buf1
    // local_dot_q7_opt(h_tm1, c->recurrent_weights->p_data, cell->units, cell->units*4, c->oshift_hw, buf2);

    // // z = z1 + z2  -> buf0
    // local_add_q7(buf1, buf2, buf0, 0, cell->units*4);
	
	// 		print_variable(buf0, "z", c->q_dec_z, cell->units*4);
	// 		print_variable(buf1, "z1", c->q_dec_z, cell->units*4);
	// 		print_variable(buf2, "z2", c->q_dec_z, cell->units*4);

    // // split the data to each gate
    // z[0] = buf0;
    // z[1] = buf0 + cell->units;
    // z[2] = buf0 + cell->units*2;
    // z[3] = buf0 + cell->units*3;

    // // i = nn.sigmoid(z0)
    // local_sigmoid_q7(z[0], cell->units, act_int_bit);
    // // f = nn.sigmoid(z1)
    // local_sigmoid_q7(z[1], cell->units, act_int_bit);
    // // o = nn.sigmoid(z3)
    // local_sigmoid_q7(z[3], cell->units, act_int_bit);
	
	// // i = nn.sigmoid(z0)
    // local_hard_sigmoid_q7(z[0], cell->units, c->q_dec_z);
    // // f = nn.sigmoid(z1)
    // local_hard_sigmoid_q7(z[1], cell->units, c->q_dec_z);
    // // o = nn.sigmoid(z3)
    // local_hard_sigmoid_q7(z[3], cell->units, c->q_dec_z);
	
	// 		print_variable(z[0], "z[0] - i", 7, cell->units);
	// 		print_variable(z[1], "z[1] - f", 7, cell->units);
	// 		print_variable(z[3], "z[3] - o", 7, cell->units);

    // /* c = f * c_tm1 + i * nn.tanh(z2) for the step 1-3. */
    // // 1. i * tanh(z2) -> buf1
    // //local_tanh_q7(z[2], cell->units, act_int_bit);
	
	// local_hard_tanh_q7(z[2], cell->units, c->q_dec_z);
	// 		print_variable(z[2], "z[2] - ?", 7, cell->units);
	
    // local_mult_q7(z[0], z[2], buf1, 14 - c->q_dec_c, cell->units); //q0.7 * q0.7 >> (shift) = q_c // i am not very sure

	// 		print_variable(buf1, "c2: i * tanh(z2) ", c->q_dec_c, cell->units);

    // // 2. f * c_tm1 -> o_state[0] 
    // local_mult_q7(z[1], c_tm1, o_state[0], c->oshift_zc, cell->units);
	
	// 		print_variable(o_state[0], "c1: f * c_tm1", c->q_dec_c, cell->units);

    // // 3. c = i*tanh + f*c_tm1 -> o_state[1]   ** fill the upper state (carry)
    // local_add_q7(buf1, o_state[0], o_state[1], 0, cell->units);
	
	// 		print_variable(o_state[1], "c = c1+c2", c->q_dec_c, cell->units);

    // /* h = o * nn.tanh(c) -> o_state[0] for the step 1-2 */
    // // 1. tanh(c) -> output_buf  --- first copy then activate. 
    // memcpy(cell->out_data, o_state[1], cell->units);
    // //
	// //local_tanh_q7(cell->out_data, cell->units, 7 - c->q_dec_c); 
	
	// local_hard_tanh_q7(cell->out_data, cell->units, c->q_dec_c); 
	
	// 		print_variable(cell->out_data, "tanh(c)", 7, cell->units);

    // // 2. h = o*tanh(c) -> o_state[0]    ** fill the lower state (memory, hidden)
    // local_mult_q7(z[3], cell->out_data, o_state[0], 7, cell->units);
	
	// 		print_variable(o_state[0], "h = o*tanh(c)", 7, cell->units);

    // // h -> output_buf ** (copy hidden to output)
    // memcpy(cell->out_data, o_state[0], cell->units);

	return NN_SUCCESS;
}
