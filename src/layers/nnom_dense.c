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

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

nnom_status_t dense_build(nnom_layer_t *layer);
nnom_status_t dense_run(nnom_layer_t *layer);

nnom_layer_t *Dense(size_t output_unit, const nnom_weight_t *w, const nnom_bias_t *b)
{
	nnom_dense_layer_t *layer;
	nnom_buf_t *comp;
	nnom_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_dense_layer_t) + sizeof(nnom_layer_io_t) * 2 + sizeof(nnom_buf_t);
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_dense_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_DENSE;
	// set buf state
	in->type = LAYER_BUF_TEMP;
	out->type = LAYER_BUF_TEMP;
	comp->type = LAYER_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	layer->super.comp = comp;
	// set run and outshape methods
	layer->super.run = dense_run;
	layer->super.build = dense_build;

	// set parameters
	layer->bias = b;
	layer->weights = w;
	layer->output_shift = w->shift;
	layer->bias_shift = b->shift; // bias is quantized to have maximum shift of weights
	layer->output_unit = output_unit;

	return (nnom_layer_t *)layer;
}

nnom_status_t dense_build(nnom_layer_t *layer)
{
	nnom_dense_layer_t *cl = (nnom_dense_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for output
	layer->out->tensor = new_tensor(NULL, 1);
	// setup new tensor
	nnom_qformat_t qfmt = {0 , 0}; // fill this later when layer API changed. 
	nnom_shape_data_t dim[1] = {cl->output_unit};
	tensor_set_attribuites(layer->out->tensor, qfmt, 1, dim);

	// vec_buffer size: dim_vec (*2, q7->q15) ? I am not sure this is right
	layer->comp->shape = shape(tensor_size(layer->in->tensor)*2, 1, 1);

	// computational cost: In * out
	layer->stat.macc = tensor_size(layer->in->tensor) * tensor_size(layer->out->tensor);
	return NN_SUCCESS;
}

nnom_status_t dense_run(nnom_layer_t *layer)
{
	nnom_status_t result = NN_SUCCESS;
	nnom_dense_layer_t *cl = (nnom_dense_layer_t *)(layer);

#if !(DENSE_WEIGHT_OPT)
	#ifdef NNOM_USING_CMSIS_NN
		result = (nnom_status_t)arm_fully_connected_q7(
	#else
		local_fully_connected_q7(
	#endif
#else
	#ifdef NNOM_USING_CMSIS_NN
		result = (nnom_status_t)arm_fully_connected_q7_opt(
	#else
		local_fully_connected_q7_opt(
	#endif
#endif
			layer->in->tensor->p_data,
			cl->weights->p_value,
			tensor_size(layer->in->tensor), layer->out->tensor->dim[0],
			cl->bias_shift, cl->output_shift,
			cl->bias->p_value,
			layer->out->tensor->p_data, (q15_t *)(layer->comp->mem->blk));


	return result;
}

