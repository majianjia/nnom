
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

nnom_status_t global_pooling_build(nnom_layer_t *layer);

nnom_layer_t *GlobalMaxPool(void)
{
	// create the normal pooling layer, the parameters are left empty to fill in later.
	// parameters will be filled in in global_pooling_build()
	nnom_layer_t *layer = MaxPool(kernel(0, 0), stride(0, 0), PADDING_VALID);

	// change to global max pool
	if (layer != NULL)
	{
		layer->type = NNOM_GLOBAL_MAXPOOL;
		layer->build = global_pooling_build;
	}

	return (nnom_layer_t *)layer;
}

nnom_layer_t *GlobalAvgPool(void)
{
	// create the normal pooling layer, the parameters are left empty to fill in later.
	// parameters will be filled in global_pooling_build() remotely
	nnom_layer_t *layer = MaxPool(kernel(0, 0), stride(0, 0), PADDING_VALID);

	// change some parameters to be recognised as avg pooling
	if (layer != NULL)
	{
		layer->type = NNOM_GLOBAL_AVGPOOL;
		layer->run = avgpool_run; // global and basic pooling share the same runner
		layer->build = global_pooling_build;
	}

	return (nnom_layer_t *)layer;
}

nnom_layer_t *GlobalSumPool(void)
{
	// create the normal pooling layer, the parameters are left empty to fill in later.
	// parameters will be filled in global_pooling_build() remotely
	nnom_layer_t *layer = MaxPool(kernel(0, 0), stride(0, 0), PADDING_VALID);

	// change some parameters to be recognised as avg pooling
	if (layer != NULL)
	{
		layer->type = NNOM_GLOBAL_SUMPOOL;
		layer->run = sumpool_run; // global and basic pooling share the same runner
		layer->build = global_pooling_build;
	}

	return (nnom_layer_t *)layer;
}

nnom_status_t global_pooling_build(nnom_layer_t *layer)
{
	nnom_maxpool_layer_t *cl = (nnom_maxpool_layer_t *)layer;
	nnom_layer_io_t *in = layer->in;
	nnom_layer_io_t *out = layer->out;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for output
	layer->out->tensor = new_tensor(NULL, 1);
	// copy then change later. 
	nnom_qformat_t qfmt = { 0 , 0 }; // fill this later when layer API changed. 
	nnom_shape_data_t dim[1] = { layer->in->tensor->dim[layer->in->tensor->num_dim-1]};
	tensor_set_attribuites(layer->out->tensor, qfmt, 1, dim);

	// different from other *_build(), the kernel..padding left by layer API needs to be set in here
	// due to the *_run() methods of global pooling are using the normall pooling's.
	// fill in the parameters left by layer APIs (GlobalAvgPool and MaxAvgPool)
	cl->kernel = shape(layer->in->tensor->dim[0], layer->in->tensor->dim[1], layer->in->tensor->dim[2]);
	cl->stride = shape(1, 1, 1);
	cl->pad = shape(0, 0, 0);
	cl->padding_type = PADDING_VALID;

	// additionally avg pooling require computational buffer, which is  2*dim_im_out*ch_im_in
	if (layer->type == NNOM_AVGPOOL || layer->type == NNOM_GLOBAL_AVGPOOL)
	{
		//  bufferA size:  2*dim_im_out*ch_im_in
		uint32_t size = layer->out->tensor->dim[0] > layer->out->tensor->dim[1] ?
							layer->out->tensor->dim[0] : layer->out->tensor->dim[1];
		layer->comp->shape = shape(2 * size * layer->in->tensor->dim[2], 1, 1);
	}
	
	// additionally sumpool
	if (layer->type == NNOM_SUMPOOL || layer->type == NNOM_GLOBAL_SUMPOOL)
		layer->comp->shape = shape(4 * tensor_size(layer->out->tensor), 1, 1);

	return NN_SUCCESS;
}


