/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-05-31     Jianjia Ma   The first version
 */


#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_layers.h"
#include "layers/nnom_conv2d_trans.h"

nnom_layer_t *conv2d_trans_s(const nnom_conv2d_config_t *config)
{
	nnom_layer_t *layer;
	layer = conv2d_s(config);
	if (layer)
	{
		layer->type = NNOM_DECONV_2D;
		layer->run = conv2d_trans_run;
		layer->build = conv2d_trans_build;
	}
	return layer;
}

nnom_layer_t *Conv2DTrans(uint32_t multiplier, nnom_3d_shape_t k, nnom_3d_shape_t s, nnom_3d_shape_t d, nnom_padding_t pad_type,
						const nnom_weight_t *w, const nnom_bias_t *b)
{
	nnom_layer_t *layer = Conv2D(multiplier, k, s, d, pad_type, w, b); // passing multiplier in .
	if (layer != NULL)
	{
		layer->type = NNOM_DECONV_2D;
		layer->run = conv2d_trans_run;
		layer->build = conv2d_trans_build;
	}
	return layer;
}

// utils, keras method
// https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/utils/conv_utils.py#L114
uint32_t conv_trans_output_length(uint32_t input_length, uint32_t kernel_size, nnom_padding_t padding, uint32_t stride_size, uint32_t dilation)
{
    uint32_t dim_size = input_length; 
    uint32_t pad;
    #define max(a,b) (a > b ? a : b)
    if (padding == PADDING_VALID)
        dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0);
    else
        dim_size = dim_size * stride_size;
    return dim_size;
}

nnom_status_t conv2d_trans_build(nnom_layer_t *layer)
{
	nnom_conv2d_trans_layer_t *cl = (nnom_conv2d_trans_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for the output
	layer->out->tensor = new_tensor(NNOM_QTYPE_PER_TENSOR, layer->in->tensor->num_dim, cl->filter_mult);
	// copy then change later. 
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);

	// now we set up the tensor shape, always HWC format
	layer->out->tensor->dim[0] = conv_trans_output_length(layer->in->tensor->dim[0], cl->kernel.h, cl->padding_type, cl->stride.h, cl->dilation.h);
	layer->out->tensor->dim[1] = conv_trans_output_length(layer->in->tensor->dim[1], cl->kernel.w, cl->padding_type, cl->stride.w, cl->dilation.w);
	layer->out->tensor->dim[2] = cl->filter_mult; // channel stays the same
	
	// fill the correct padding
	if(cl->padding_type == PADDING_SAME)
	{	
		cl->pad.h =  cl->kernel.h/2;
		cl->pad.w =  cl->kernel.w/2;
//		cl->pad.h = ((cl->stride.h - 1)*layer->in->tensor->dim[0] - cl->stride.h + cl->kernel.h)/2;
//		cl->pad.w = ((cl->stride.w - 1)*layer->in->tensor->dim[1] - cl->stride.w + cl->kernel.w)/2;
		cl->pad.c = 0;
	}
	else
	{
		cl->pad.h = 0;
		cl->pad.w = 0;
		cl->pad.c = 0;
	}

	// bufferA size: (1D shape)
	// 2*ch_im_in*dim_kernel*dim_kernel
	//layer->comp->size = 2 * 2 * layer->in->tensor->dim[2] * cl->kernel.w * cl->kernel.h;
	// computational cost: K x K x Cin x Hour x Wout x Cout
	layer->stat.macc = cl->kernel.w * cl->kernel.h * layer->in->tensor->dim[2] * tensor_size(layer->out->tensor);
	return NN_SUCCESS;
}


nnom_status_t conv2d_trans_run(nnom_layer_t *layer)
{
    nnom_conv2d_trans_layer_t *cl = (nnom_conv2d_trans_layer_t *)layer;
	nnom_qformat_param_t bias_shift = cl->bias_lshift[0];			// this is not correct but a temporary fix solution for backward compatibility.
	nnom_qformat_param_t output_shift = cl->output_rshift[0];
	local_conv_trans_HWC_q7_nonsquare(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weight->p_data, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h, cl->dilation.w, cl->dilation.h,
				cl->bias->p_data, bias_shift, output_shift,
				layer->out->tensor->p_data,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0], NULL, NULL);
	return NN_SUCCESS;
}


