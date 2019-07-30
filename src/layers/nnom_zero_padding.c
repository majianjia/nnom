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

nnom_status_t zero_padding_build(nnom_layer_t *layer);
nnom_status_t zero_padding_run(nnom_layer_t *layer);

// Zero padding layer
nnom_layer_t *ZeroPadding(nnom_border_t pad)
{
	nnom_zero_padding_layer_t *layer;
	nnom_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_zero_padding_layer_t) + sizeof(nnom_layer_io_t) * 2;
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_zero_padding_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_ZERO_PADDING;
	// set buf state
	in->type = LAYER_BUF_TEMP;
	out->type = LAYER_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	// set run and outshape methods
	layer->super.run = zero_padding_run;
	layer->super.build = zero_padding_build;

	// set parameters
	layer->pad = pad;
	
	return (nnom_layer_t*)layer;
}

nnom_status_t zero_padding_build(nnom_layer_t* layer)
{
	nnom_zero_padding_layer_t *cl = (nnom_zero_padding_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for output
	layer->out->tensor = new_tensor(NULL, layer->in->tensor->num_dim);
	// copy then change later. 
	tensor_cpy_attributes(layer->out->tensor, layer->in->tensor);

	// output shape
	layer->out->tensor->dim[1] = layer->in->tensor->dim[1] + cl->pad.left + cl->pad.right;
	layer->out->tensor->dim[0] = layer->in->tensor->dim[0] + cl->pad.top + cl->pad.bottom;
	layer->out->tensor->dim[2] = layer->in->tensor->dim[2];
	return NN_SUCCESS;
}

nnom_status_t zero_padding_run(nnom_layer_t * layer)
{
	nnom_zero_padding_layer_t *cl = (nnom_zero_padding_layer_t*)layer;
	
#ifdef NNOM_USING_CHW
	local_zero_padding_CHW_q7(
#else
	local_zero_padding_HWC_q7(
#endif
						layer->in->mem->blk, 
						layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
						cl->pad.top,
						cl->pad.bottom,
						cl->pad.left,
						cl->pad.right,
						layer->out->mem->blk,
						layer->out->tensor->dim[1], layer->out->tensor->dim[0]);

	return NN_SUCCESS;
}

