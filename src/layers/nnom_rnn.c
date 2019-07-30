/*
 * Copyright (c) 2018-2019
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-07-23     Jianjia Ma   The first version
 */

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_layers.h"

nnom_status_t rnn_build(nnom_layer_t *layer);
nnom_status_t rnn_run(nnom_layer_t *layer);

// Simple RNN
// unit = output shape
// type of activation
nnom_rnn_cell_t *SimpleCell(size_t units, nnom_activation_t *activation, const nnom_weight_t *w, const nnom_bias_t *b)
{
	nnom_simple_rnn_cell_t *cell;
	cell = nnom_mem(sizeof(nnom_simple_rnn_cell_t));
	if (cell == NULL)
		return (nnom_rnn_cell_t *)cell;
	// set parameters
	cell->activation = activation;
	cell->super.units = units;
	cell->super.run = cell_simple_rnn_run;

	cell->bias = b;
	cell->weights = w;
	//cell->output_shift = w->shift;
	//cell->bias_shift = w->shift - b->shift;	// bias is quantized to have maximum shift of weights

	return (nnom_rnn_cell_t *)cell;
}

// RNN
nnom_layer_t *RNN(nnom_rnn_cell_t *cell, bool return_sequence)
{

	nnom_rnn_layer_t *layer;
	nnom_buf_t *comp;
	nnom_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_rnn_layer_t) + sizeof(nnom_layer_io_t) * 2 + sizeof(nnom_buf_t);
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_rnn_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_RNN;
	// set buf state
	in->type = LAYER_BUF_TEMP;
	out->type = LAYER_BUF_TEMP;
	comp->type = LAYER_BUF_RESERVED; // reserve buf for RNN state (statfulness)
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	layer->super.comp = comp;
	// set run and outshape methods
	layer->super.run = rnn_run;
	layer->super.build = rnn_build;

	// rnn parameters.
	layer->return_sequence = return_sequence;
	layer->cell = cell;

	return (nnom_layer_t *)layer;
}


// the state buffer and computational buffer shape of the cell
nnom_status_t simplecell_build(nnom_layer_t* layer, nnom_rnn_cell_t* cell)
{

	return NN_SUCCESS;
}

// TODO
nnom_status_t rnn_build(nnom_layer_t* layer)
{
	nnom_rnn_layer_t* cl = (nnom_rnn_layer_t*)layer;
	/*
	// get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;

	// output shape
	//layer->out->shape = layer->in->shape;

	// TODO
	// calculate computational and stat buf size
	layer->comp = 0;

	// calculate output shape according to the return_sequence
	if (cl->return_sequence)
	{
		layer->out->shape.h = 1;				  // batch?
		layer->out->shape.w = layer->in->shape.w; // timestamp (same timestamps)
		layer->out->shape.c = cl->cell->units;	 // output unit
	}
	else
	{
		layer->out->shape.h = 1;			  // batch?
		layer->out->shape.w = 1;			  // timestamp
		layer->out->shape.c = cl->cell->units; // output unit
	}
	*/
	return NN_SUCCESS;
}

nnom_status_t cell_simple_rnn_run(nnom_layer_t *layer)
{
	/*
	nnom_status_t result;
	// cell / layer
	nnom_rnn_layer_t* cl 	= (nnom_rnn_layer_t *)layer;
	nnom_simple_rnn_cell_t* cell = (nnom_simple_rnn_cell_t*)cl->cell;
	// parameters
	size_t input_size 		= layer->in->shape.c;				// in rnn, h = 1, w = timestamp, c = feature size. 
	size_t output_size 		= cell->super.units;					// output size = state size in keras. 
	q7_t* weight 			= (q7_t*)cell->weights->p_value;
	q7_t* re_weight 		= (q7_t*)cell->weights->p_value + input_size;
	q7_t* bias				= (q7_t*)cell->bias->p_value;
	q7_t* bias_dummy		= (q7_t*)cell->bias->p_value + output_size;// this must be a dummy bias for all zero. 
	uint16_t bias_shift 	= cell->bias->shift; 			 	// not sure
	uint16_t output_shift 	= cell->weights->shift; 			// not correct
	uint8_t* vector_buf 	= layer->comp->mem->blk;			// not correct, buf for calculation
	
	// layer->comp buf is use to store states and intermmediate buffer
	// state buf | B1	|compute buf;  Additionaly, cell->output buffer can be used for calulation
	// 
	// h = tanh or relu(w*x + b_dummy + h*x + bias)

	// w*x + b_dummy
	// buff: input -> B1
	result = (nnom_status_t)arm_fully_connected_q7(
		cell->super.input_buf,
		weight,
		input_size, output_size,
		bias_shift, output_shift,
		bias_dummy,
		cell->super.output_buf, (q15_t*)vector_buf);
	
	// h*x + bias (paramters are wrong)
	// buff: state -> output
	result = (nnom_status_t)arm_fully_connected_q7(
		cell->super.input_buf,
		re_weight,
		input_size, output_size,
		bias_shift, output_shift,
		bias,
		cell->super.output_buf, (q15_t*)vector_buf);
	
	// add (paramters are wrong)
	// buff: B1 + output -> state 
	arm_add_q7(layer->in->mem->blk, layer->out->mem->blk, layer->out->mem->blk, output_size);
	
	// finally the activation (thinking of changing the activation's run interfaces. )
	// buff: state
	result = act_direct_run(layer, cell->activation,  cell->super.output_buf, output_size, layer->in->qfmt);

	// copy to output
	//memcpy(cell->super.output_buf, state, output_size);

	*/
	return NN_SUCCESS;
}


nnom_status_t rnn_run(nnom_layer_t* layer)
{
	nnom_status_t result;
	/*
	nnom_rnn_layer_t* cl = (nnom_rnn_layer_t*)(layer);
	size_t timestamps_size = layer->in->shape.w;
	size_t feature_size = layer->in->shape.c;
	size_t output_size = cl->cell->units;

	// set the state buffer
	cl->cell->state_buf = layer->comp->mem;

	// currently not support stateful. and not support reserved mem block
	if (!cl->stateful)
		memset(cl->cell->state_buf, 0, shape_size(&layer->comp->shape));

	// run
	for (uint32_t round = 0; round < timestamps_size; round++)
	{
		// set input buffer
		cl->cell->input_buf = (q7_t*)layer->in->mem->blk + feature_size * round;
		if (cl->return_sequence)
			cl->cell->output_buf = (q7_t*)layer->out->mem->blk + output_size * round;
		else
			cl->cell->output_buf = layer->out->mem->blk;

		// run it
		result = cl->cell->run(layer);
	}
	*/
	return result;
}

