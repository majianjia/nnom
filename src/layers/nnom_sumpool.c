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
#include "layers/nnom_sumpool.h"

nnom_status_t sumpool_build(nnom_layer_t *layer);
nnom_status_t sumpool_run(nnom_layer_t *layer);

nnom_layer_t *SumPool(nnom_3d_shape_t k, nnom_3d_shape_t s, nnom_padding_t pad_type)
{
	nnom_layer_t *layer = MaxPool(k, s, pad_type);

	if (layer != NULL)
	{
		layer->type = NNOM_SUMPOOL;
		layer->run = sumpool_run;
		layer->build = sumpool_build;
	}
	return (nnom_layer_t *)layer;
}


nnom_status_t sumpool_build(nnom_layer_t *layer)
{
	// avg pooling share the same output shape, stride, padding setting.
	maxpool_build(layer);

	// however, avg pooling require a computational buffer.
	layer->comp->size = 4 * tensor_size(layer->out->tensor);

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
			layer->in->tensor->p_data, 				
			layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
			cl->kernel.w, cl->kernel.h, 
			cl->pad.w, cl->pad.h,
			cl->stride.w, cl->stride.h,
			layer->out->tensor->dim[1], layer->out->tensor->dim[0],
			layer->comp->mem->blk,
			layer->out->tensor->p_data);

	return NN_SUCCESS;
}
