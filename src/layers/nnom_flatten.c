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

nnom_status_t flatten_build(nnom_layer_t *layer);
nnom_status_t flatten_run(nnom_layer_t *layer);

nnom_layer_t *Flatten(void)
{
	nnom_layer_t *layer;
	nnom_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_layer_t) + sizeof(nnom_layer_io_t) * 2;
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->type = NNOM_FLATTEN;
	layer->run = flatten_run;
	layer->build = flatten_build;
	// set buf state
	in->type = LAYER_BUF_TEMP;
	#ifdef NNOM_USING_CHW
		out->type = LAYER_BUF_TEMP; // test for CHW format
	#else
		out->type = LAYER_BUF_NULL; 
	#endif
	// put in & out on the layer.
	layer->in = io_init(layer, in);
	layer->out = io_init(layer, out);

	return layer;
}

nnom_status_t flatten_build(nnom_layer_t *layer)
{ // get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;

	// output shape
	layer->out->shape.h = layer->in->shape.h * layer->in->shape.w * layer->in->shape.c;
	layer->out->shape.w = 1;
	layer->out->shape.c = 1;

	return NN_SUCCESS;
}

nnom_status_t flatten_run(nnom_layer_t *layer)
{
	#ifdef NNOM_USING_CHW
	// CHW format must reorder to HWC for dense layer and all other 1D layer (?)
	chw2hwc_q7(layer->in->shape, layer->in->mem->blk, layer->out->mem->blk);
	#endif
	// you must be kidding me
	return NN_SUCCESS;
}
