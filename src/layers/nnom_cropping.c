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

nnom_status_t cropping_build(nnom_layer_t *layer);
nnom_status_t cropping_run(nnom_layer_t *layer);

// Cropping layer
nnom_layer_t *Cropping(nnom_border_t pad)
{
	nnom_layer_t *layer;
	// most setting are the same as zero padding
	layer = ZeroPadding(pad);
	
	layer->type = NNOM_CROPPING;
	layer->run = cropping_run;
	layer->build = cropping_build;

	return layer;
}

nnom_status_t cropping_build(nnom_layer_t* layer)
{
	nnom_cropping_layer_t *cl = (nnom_cropping_layer_t *)layer;
	
	// get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;
	
	// output shape
	if(layer->in->shape.w <= (cl->pad.left + cl->pad.right) || 
		layer->in->shape.h <= (cl->pad.top + cl->pad.bottom))
		return NN_ARGUMENT_ERROR;
	
	layer->out->shape.w = layer->in->shape.w - (cl->pad.left + cl->pad.right);
	layer->out->shape.h = layer->in->shape.h - (cl->pad.top + cl->pad.bottom);
	layer->out->shape.c = layer->in->shape.c;
	return NN_SUCCESS;
}


nnom_status_t cropping_run(nnom_layer_t * layer)
{
	nnom_cropping_layer_t *cl = (nnom_cropping_layer_t*)layer;
	
#ifdef NNOM_USING_CHW
	local_cropping_CHW_q7(
#else
	local_cropping_HWC_q7(
#endif	
						layer->in->mem->blk, 
						layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
						cl->pad.top,
						cl->pad.bottom,
						cl->pad.left,
						cl->pad.right,
						layer->out->mem->blk,
						layer->out->shape.w, layer->out->shape.h);

	return NN_SUCCESS;
}
