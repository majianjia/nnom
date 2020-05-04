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
#include "layers/nnom_input.h"

nnom_status_t input_build(nnom_layer_t *layer);
nnom_status_t input_run(nnom_layer_t *layer);


nnom_layer_t *Input(nnom_3d_shape_t input_shape, void *p_buf)
{
	nnom_io_layer_t *layer;
	nnom_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	layer = nnom_mem(sizeof(nnom_io_layer_t) + sizeof(nnom_layer_io_t) * 2);
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
	in->type = NNOM_TENSOR_BUF_TEMP;
	out->type = NNOM_TENSOR_BUF_NULL;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);

	// set parameters
	layer->shape = input_shape;
	layer->buf = p_buf;

	// experimental: fixed input dim to 3
	// input normally dont have a tensor, so we create one to store the initial data. 
	nnom_shape_data_t dim[3] = { input_shape.h, input_shape.w, input_shape.c };
	layer->super.in->tensor = new_tensor(NNOM_QTYPE_PER_TENSOR, 3, input_shape.c);
	tensor_set_attr_v(layer->super.in->tensor, 7, 0, dim, sizeof(dim)/sizeof(nnom_shape_data_t), 8);
	return (nnom_layer_t *)layer;
}

nnom_status_t input_build(nnom_layer_t* layer)
{
	// the input tensor of inputlayer has assigned previously 

	// output tensor
	// 1. allocate a new tensor for output
	// 2. set the same dim, qfmt to the new tensor.
	layer->out->tensor = new_tensor(NNOM_QTYPE_PER_TENSOR, layer->in->tensor->num_dim, tensor_get_num_channel(layer->in->tensor));
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);

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
