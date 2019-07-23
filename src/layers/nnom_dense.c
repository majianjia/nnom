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
	nnom_layer_io_t *in = layer->in;
	nnom_layer_io_t *out = layer->out;

	//get the last layer's output as input shape
	in->shape = in->hook.io->shape;

	// incase the input hasnt been flattened.
	in->shape.h = in->shape.h * in->shape.w * in->shape.c;
	in->shape.w = 1;
	in->shape.c = 1;

	out->shape.h = cl->output_unit;
	out->shape.w = 1;
	out->shape.c = 1;

	// vec_buffer size: dim_vec (*2, q7->q15)
	layer->comp->shape = shape(shape_size(&in->shape)*2, 1, 1);

	// computational cost: In * out
	layer->stat.macc = in->shape.h * out->shape.h;
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
			layer->in->mem->blk,
			cl->weights->p_value,
			layer->in->shape.h, layer->out->shape.h,
			cl->bias_shift, cl->output_shift,
			cl->bias->p_value,
			layer->out->mem->blk, (q15_t *)(layer->comp->mem->blk));
#else
	#ifdef NNOM_USING_CMSIS_NN
		result = (nnom_status_t)arm_fully_connected_q7_opt(
	#else
		local_fully_connected_q7_opt(
	#endif
			layer->in->mem->blk,
			cl->weights->p_value,
			layer->in->shape.h, layer->out->shape.h,
			cl->bias_shift, cl->output_shift,
			cl->bias->p_value,
			layer->out->mem->blk, (q15_t *)(layer->comp->mem->blk));
#endif

	return result;
}

