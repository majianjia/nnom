/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-12-07     Jianjia Ma   The first version
 */

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_layers.h"
#include "layers/nnom_reshape.h"


nnom_layer_t *reshape_s(const nnom_reshape_config_t *config)
{
	nnom_reshape_layer_t *layer;
	nnom_layer_io_t *in, *out;
    
	// allocate a block memory for all the sub handles and shifts.
	size_t mem_size = sizeof(nnom_reshape_layer_t) + sizeof(nnom_layer_io_t) * 2 ;
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;
	
	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_reshape_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_RESHAPE;
	layer->super.run = reshape_run;
	layer->super.build = reshape_build;
	// set buf state
	in->type = NNOM_TENSOR_BUF_TEMP;
	out->type = NNOM_TENSOR_BUF_NULL; 

    // config
    //nnom_memcpy(layer->dim, config->dim, config->num_dim * sizeof(nnom_shape_data_t));
	layer->super.config = (void*)config;
    layer->dim = config->dim;		// temporary use the config directly. (not preferable.) 
	layer->num_dim = config->num_dim;

	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);

	return (nnom_layer_t *)layer;
}

nnom_status_t reshape_build(nnom_layer_t *layer)
{
	nnom_reshape_layer_t *cl = (nnom_reshape_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for output
	layer->out->tensor = new_tensor(NNOM_QTYPE_PER_TENSOR, cl->num_dim, cl->dim[cl->num_dim-1]);
	tensor_set_attr(layer->out->tensor, layer->in->tensor->q_dec, layer->in->tensor->q_offset, cl->dim, cl->num_dim, 8);

	return NN_SUCCESS;
}

nnom_status_t reshape_run(nnom_layer_t *layer)
{
	return NN_SUCCESS;
}

