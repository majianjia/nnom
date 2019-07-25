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

nnom_status_t upsample_build(nnom_layer_t *layer);
nnom_status_t upsample_run(nnom_layer_t *layer);

// up sampling layer
nnom_layer_t *UpSample(nnom_shape_t kernel)
{
	nnom_upsample_layer_t *layer;
	nnom_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_upsample_layer_t) + sizeof(nnom_layer_io_t) * 2;
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_upsample_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_UPSAMPLE;
	// set buf state
	in->type = LAYER_BUF_TEMP;
	out->type = LAYER_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	// set run and outshape methods
	layer->super.run = upsample_run;
	layer->super.build = upsample_build;

	// set parameters
	layer->kernel = kernel;
	
	return (nnom_layer_t*)layer;
}

nnom_status_t upsample_build(nnom_layer_t *layer)
{
	nnom_upsample_layer_t* cl = (nnom_upsample_layer_t*)layer;

	// get the last layer's output as input shape
	layer->in->tensor = layer->in->hook.io->tensor;
	// output tensor
	// 1. allocate a new tensor for output
	// 2. set the same dim, qfmt to the new tensor.
	layer->out->tensor = new_tensor(NULL, layer->in->tensor->num_dim);
	tensor_cpy_attributes(layer->out->tensor, layer->in->tensor);

	// enlarge w and h, c stay the same.
	layer->out->tensor->dim[0] = layer->in->tensor->dim[0] * cl->kernel.h;
	layer->out->tensor->dim[1] = layer->in->tensor->dim[1] * cl->kernel.w;

	return NN_SUCCESS;
}

// up sampling, or so called unpooling
nnom_status_t upsample_run(nnom_layer_t *layer)
{
	nnom_upsample_layer_t *cl = (nnom_upsample_layer_t *)(layer);
#ifdef NNOM_USING_CHW
	local_up_sampling_q7_CHW(				
#else
	local_up_sampling_q7_HWC(
#endif
			layer->in->mem->blk, 				
			layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
			cl->kernel.w, cl->kernel.h, 
			layer->out->tensor->dim[1], layer->out->tensor->dim[0],
			NULL,
			layer->out->mem->blk);
	return NN_SUCCESS;
}
