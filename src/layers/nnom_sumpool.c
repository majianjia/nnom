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

nnom_status_t sumpooling_build(nnom_layer_t *layer);
nnom_status_t sumpool_run(nnom_layer_t *layer);

nnom_layer_t *SumPool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad_type)
{
	nnom_layer_t *layer = MaxPool(k, s, pad_type);

	if (layer != NULL)
	{
		layer->type = NNOM_SUMPOOL;
		layer->run = sumpool_run;
		layer->build = sumpooling_build;
	}
	return (nnom_layer_t *)layer;
}


nnom_status_t sumpooling_build(nnom_layer_t *layer)
{
	// avg pooling share the same output shape, stride, padding setting.
	maxpooling_build(layer);

	// however, avg pooling require a computational buffer.
	layer->comp->shape = shape(4 * layer->out->shape.h * layer->out->shape.w * layer->out->shape.c, 1, 1);

	return NN_SUCCESS;
}


// sum pooling, dynamic change Q format, must be used in the last layer before softmax in current version
nnom_status_t sumpool_run(nnom_layer_t *layer)
{
	nnom_sumpool_layer_t *cl = (nnom_sumpool_layer_t *)(layer);
	
#ifdef NNOM_USING_CHW
	local_sumpool_q7_CHW(				
#else
	local_sumpool_q7_HWC(
#endif
			layer->in->mem->blk, 				
			layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
			cl->kernel.w, cl->kernel.h, 
			cl->pad.w, cl->pad.h,
			cl->stride.w, cl->stride.h,
			layer->out->shape.w, layer->out->shape.h,
			layer->comp->mem->blk,
			layer->out->mem->blk);

	return NN_SUCCESS;
}
