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

nnom_status_t input_build(nnom_layer_t *layer);
nnom_status_t input_run(nnom_layer_t *layer);


nnom_layer_t *Input(nnom_shape_t input_shape, void *p_buf)
{
	nnom_io_layer_t *layer;
	nnom_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_io_layer_t) + sizeof(nnom_layer_io_t) * 2;
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_io_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_INPUT;
	layer->super.run = input_run;
	layer->super.build = input_build;
	// set buf state
	in->type = LAYER_BUF_TEMP;
	out->type = LAYER_BUF_NULL;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);

	// set parameters
	layer->shape = input_shape;
	layer->buf = p_buf;

	// experimental: fixed input dim to 3
	// input normally dont have a tensor, so we create one to store the initial data. 
	nnom_shape_data_t dim[3] = { input_shape.h, input_shape.w, input_shape.c };
	layer->super.in->tensor = new_tensor(NULL, 3); 
	tensor_set_attribuites(layer->super.in->tensor, layer->super.in->tensor->qfmt, 3, dim);

	return (nnom_layer_t *)layer;
}

nnom_status_t input_build(nnom_layer_t* layer)
{
	// the input tensor of inputlayer has assigned previously 

	// output tensor
	// 1. allocate a new tensor for output
	// 2. set the same dim, qfmt to the new tensor.
	layer->out->tensor = new_tensor(NULL, layer->in->tensor->num_dim);
	tensor_cpy_attributes(layer->out->tensor, layer->in->tensor);

	// now this build has passed the input tensors (shapes, formats) to the new tensors. 
	return NN_SUCCESS;
}


nnom_status_t input_run(nnom_layer_t *layer)
{
	nnom_io_layer_t *cl = (nnom_io_layer_t *)layer;
#ifdef NNOM_USING_CHW
	tensor_hwc2chw_q7(layer->out->tensor, layer->in->tensor);
#else
	memcpy(layer->in->tensor->p_data, cl->buf, tensor_size(layer->in->tensor));
#endif
	return NN_SUCCESS;
}
