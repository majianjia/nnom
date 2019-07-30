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

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

nnom_status_t avgpooling_build(nnom_layer_t *layer);
nnom_status_t avgpool_run(nnom_layer_t *layer);

nnom_layer_t *AvgPool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad_type)
{
	nnom_layer_t *layer = MaxPool(k, s, pad_type);

	if (layer != NULL)
	{
		layer->type = NNOM_AVGPOOL;
		layer->run = avgpool_run;
		layer->build = avgpooling_build;
	}
	return (nnom_layer_t *)layer;
}

nnom_status_t avgpooling_build(nnom_layer_t *layer)
{
	uint32_t size;
	// avg pooling share the same output shape, stride, padding setting.
	maxpooling_build(layer);

	#ifdef NNOM_USING_CMSIS_NN
	// however, avg pooling require a computational buffer.
	//  bufferA size:  2*dim_im_out*ch_im_in
	size = layer->out->tensor->dim[1] > layer->out->tensor->dim[0] ?
						layer->out->tensor->dim[1] : layer->out->tensor->dim[0];
	layer->comp->shape = shape(2 * size * layer->in->tensor->dim[2], 1, 1);
	#endif

	return NN_SUCCESS;
}

nnom_status_t avgpool_run(nnom_layer_t *layer)
{
	nnom_avgpool_layer_t *cl = (nnom_avgpool_layer_t *)(layer);
	
#ifdef NNOM_USING_CHW
	local_avepool_q7_CHW(layer->in->tensor->p_data, 				
			layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
			cl->kernel.w, cl->kernel.h, 
			cl->pad.w, cl->pad.h,
			cl->stride.w, cl->stride.h,
			layer->out->tensor->dim[1], layer->out->tensor->dim[0],
			cl->output_shift,
			NULL,
			layer->out->tensor->p_data);
#else //end of CHW
	#ifdef NNOM_USING_CMSIS_NN
	// 2D, square
	if (layer->in->tensor->dim[1] == layer->in->tensor->dim[0] &&
		layer->out->tensor->dim[1] == layer->out->tensor->dim[0] &&
		cl->output_shift == 0)
	{
		arm_avepool_q7_HWC(
			layer->in->tensor->p_data,
			layer->in->tensor->dim[1], layer->in->tensor->dim[2],
			cl->kernel.w, cl->pad.w, cl->stride.w,
			layer->out->tensor->dim[1],
			layer->comp->mem->blk,
			layer->out->tensor->p_data);
	}
	// none square 2D, or 1D
	else
	#endif
	{
		// CMSIS-NN does not support none-square pooling, we have to use local implementation
		local_avepool_q7_HWC(layer->in->tensor->p_data, 				
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, 
				cl->pad.w, cl->pad.h,
				cl->stride.w, cl->stride.h,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0],
				cl->output_shift,
				NULL,
				layer->out->tensor->p_data);
	}
#endif
	return NN_SUCCESS;
}
