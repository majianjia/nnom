/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
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
#include "layers/nnom_dw_conv2d.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

nnom_layer_t *dw_conv2d_s(nnom_conv2d_config_t *config)
{
	nnom_layer_t *layer;
	layer = conv2d_s(config);
	if (layer)
	{
		layer->type = NNOM_DW_CONV_2D;
		layer->run = dw_conv2d_run;
		layer->build = dw_conv2d_build;
	}
	return layer;
}

nnom_layer_t *DW_Conv2D(uint32_t multiplier, nnom_3d_shape_t k, nnom_3d_shape_t s, nnom_3d_shape_t d, nnom_padding_t pad_type,
						const nnom_weight_t *w, const nnom_bias_t *b)
{
	nnom_layer_t *layer = Conv2D(multiplier, k, s, d, pad_type, w, b); // passing multiplier in .
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
	layer->out->tensor = new_tensor(NNOM_QTYPE_PER_TENSOR, layer->in->tensor->num_dim, tensor_get_num_channel(layer->in->tensor) * cl->filter_mult);
	// copy then change later. 
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);

	// now we set up the tensor shape, always HWC format
	layer->out->tensor->dim[0] = conv_output_length(layer->in->tensor->dim[0], cl->kernel.h, cl->padding_type, cl->stride.h, cl->dilation.h);
	layer->out->tensor->dim[1] = conv_output_length(layer->in->tensor->dim[1], cl->kernel.w, cl->padding_type, cl->stride.w, cl->dilation.w);
	layer->out->tensor->dim[2] = layer->in->tensor->dim[2] * cl->filter_mult; // channel stays the same

	// bufferA size: (1D shape)
	layer->comp->size = 2 * 2 * (layer->in->tensor->dim[2] / cl->filter_mult) * cl->kernel.w * cl->kernel.h;

	// computational cost: K x K x Cin x Hout x Wout x Multiplier
	// or                : K x K x Cout x Hout x Wout
	layer->stat.macc = cl->kernel.w * cl->kernel.h * tensor_size(layer->out->tensor);
	return NN_SUCCESS;
}

nnom_status_t dw_conv2d_run(nnom_layer_t *layer)
{
	nnom_status_t result = NN_SUCCESS;
	nnom_conv2d_layer_t *cl = (nnom_conv2d_layer_t *)layer;

	nnom_qformat_param_t bias_shift = cl->bias->q_dec[0];				//
	nnom_qformat_param_t output_shift = cl->weight->q_dec[0];			// need to be changed later. 

#ifdef NNOM_USING_CHW
	local_depthwise_separable_conv_CHW_q7_nonsquare(
#else	
	#ifdef NNOM_USING_CMSIS_NN
	// Current CMSIS-NN does not support dilation
	if(cl->dilation.w ==1 && cl->dilation.h == 1)
	{
		// CMSIS-NN only support 1 mulplipier in depthwise conv
		if (cl->filter_mult != 1 || layer->in->tensor->dim[2] % 2 != 0 || layer->out->tensor->dim[2] % 2)
			return NN_ARGUMENT_ERROR;
		result = (nnom_status_t)arm_depthwise_separable_conv_HWC_q7_nonsquare(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weight->p_data,
				layer->in->tensor->dim[2],
				cl->kernel.w, cl->kernel.h,
				cl->pad.w, cl->pad.h,
				cl->stride.w, cl->stride.h,
				cl->bias->p_data,
				bias_shift, output_shift,
				layer->out->tensor->p_data,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0], (q15_t *)(layer->comp->mem->blk), NULL);
	}
	else
	#endif // NNOM_USING_CMSIS_NN
	local_depthwise_separable_conv_HWC_q7_nonsquare(
#endif
		layer->in->tensor->p_data,
		layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
		cl->weight->p_data,
		layer->in->tensor->dim[2],
		cl->kernel.w, cl->kernel.h,
		cl->pad.w, cl->pad.h,
		cl->stride.w, cl->stride.h,
		cl->dilation.w, cl->dilation.h,
		cl->bias->p_data,
		bias_shift, output_shift,
		layer->out->tensor->p_data,
		layer->out->tensor->dim[1], layer->out->tensor->dim[0], (q15_t *)(layer->comp->mem->blk), NULL);

	return result;
}
