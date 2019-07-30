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

nnom_status_t dw_conv2d_build(nnom_layer_t *layer);
nnom_status_t dw_conv2d_run(nnom_layer_t *layer);


nnom_layer_t *DW_Conv2D(uint32_t multiplier, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad_type,
						const nnom_weight_t *w, const nnom_bias_t *b)
{
	nnom_layer_t *layer = Conv2D(multiplier, k, s, pad_type, w, b); // passing multiplier in .
	if (layer != NULL)
	{
		layer->type = NNOM_DW_CONV_2D;
		layer->run = dw_conv2d_run;
		layer->build = dw_conv2d_build;
	}
	return layer;
}

nnom_status_t dw_conv2d_build(nnom_layer_t *layer)
{
	nnom_conv2d_layer_t *cl = (nnom_conv2d_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for output
	layer->out->tensor = new_tensor(NULL, layer->in->tensor->num_dim);
	// copy then change later. 
	tensor_cpy_attributes(layer->out->tensor, layer->in->tensor);

	// now we set up the tensor shape, always HWC format
	if (cl->padding_type == PADDING_SAME)
	{
		layer->out->tensor->dim[0] = NN_CEILIF(layer->in->tensor->dim[0], cl->stride.h);
		layer->out->tensor->dim[1] = NN_CEILIF(layer->in->tensor->dim[1], cl->stride.w);
		layer->out->tensor->dim[2] = layer->in->tensor->dim[2] * cl->filter_mult; // channel stays the same
	}
	else
	{
		layer->out->tensor->dim[0] = NN_CEILIF(layer->in->tensor->dim[0] - cl->kernel.h + 1, cl->stride.h);
		layer->out->tensor->dim[1] = NN_CEILIF(layer->in->tensor->dim[1] - cl->kernel.w + 1, cl->stride.w);
		layer->out->tensor->dim[2] = layer->in->tensor->dim[2] * cl->filter_mult;
	}

	// bufferA size: (1D shape)
	layer->comp->shape = shape(2 * 2 * (layer->in->tensor->dim[2] / cl->filter_mult) * cl->kernel.w * cl->kernel.h, 1, 1);

	// computational cost: K x K x Cin x Hout x Wout x Multiplier
	// or                : K x K x Cout x Hout x Wout
	layer->stat.macc = cl->kernel.w * cl->kernel.h * tensor_size(layer->out->tensor);
	return NN_SUCCESS;
}

nnom_status_t dw_conv2d_run(nnom_layer_t *layer)
{
	nnom_status_t result = NN_SUCCESS;
	nnom_conv2d_layer_t *cl = (nnom_conv2d_layer_t *)layer;

#ifdef NNOM_USING_CHW
	local_depthwise_separable_conv_CHW_q7_nonsquare(
#else	
	#ifdef NNOM_USING_CMSIS_NN
		// CMSIS-NN only support 1 mulplipier in depthwise conv
		if (cl->filter_mult != 1 || layer->in->tensor->dim[2] % 2 != 0 || layer->out->tensor->dim[2] % 2)
			return NN_ARGUMENT_ERROR;
		result = (nnom_status_t)arm_depthwise_separable_conv_HWC_q7_nonsquare(
	#else
		local_depthwise_separable_conv_HWC_q7_nonsquare(
	#endif
#endif
		layer->in->tensor->p_data,
		layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
		cl->weights->p_value,
		layer->in->tensor->dim[2],
		cl->kernel.w, cl->kernel.h,
		cl->pad.w, cl->pad.h,
		cl->stride.w, cl->stride.h,
		cl->bias->p_value,
		cl->bias_shift, cl->output_shift,
		layer->out->tensor->p_data,
		layer->out->tensor->dim[1], layer->out->tensor->dim[0], (q15_t *)(layer->comp->mem->blk), NULL);

	return result;
}
