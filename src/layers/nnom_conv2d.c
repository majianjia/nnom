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

nnom_status_t conv2d_run(nnom_layer_t *layer);
nnom_status_t conv2d_build(nnom_layer_t *layer);

// Conv2D
// multiplier of (output/input channel),
// shape of kernal, shape of strides, weight struct, bias struct
nnom_layer_t *Conv2D(uint32_t filters, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad_type,
					 const nnom_weight_t *w, const nnom_bias_t *b)
{
	nnom_conv2d_layer_t *layer;
	nnom_buf_t *comp;
	nnom_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_conv2d_layer_t) + sizeof(nnom_layer_io_t) * 2 + sizeof(nnom_buf_t);
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_conv2d_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_CONV_2D;
	// set buf state
	in->type = LAYER_BUF_TEMP;
	out->type = LAYER_BUF_TEMP;
	comp->type = LAYER_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	layer->super.comp = comp;
	// set run method & output shape
	layer->super.run = conv2d_run;
	layer->super.build = conv2d_build;

	// get the private parameters
	layer->kernel = k;
	layer->stride = s;
	layer->bias = b;
	layer->weights = w;
	layer->output_shift = w->shift;
	layer->bias_shift = b->shift; // bias is quantized to have maximum shift of weights
	layer->filter_mult = filters; // for convs, this means filter number
	layer->padding_type = pad_type;

	// padding
	if (layer->padding_type == PADDING_SAME)
	{
		layer->pad.w = (k.w - 1) / 2;
		layer->pad.h = (k.h - 1) / 2;
		layer->pad.c = (k.c - 1) / 2;
	}

	return (nnom_layer_t *)layer;
}



nnom_status_t conv2d_build(nnom_layer_t *layer)
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
		layer->out->tensor->dim[2] = cl->filter_mult; // channel stays the same
	}
	else
	{
		layer->out->tensor->dim[0] = NN_CEILIF(layer->in->tensor->dim[0] - cl->kernel.h + 1, cl->stride.h);
		layer->out->tensor->dim[1] = NN_CEILIF(layer->in->tensor->dim[1] - cl->kernel.w + 1, cl->stride.w);
		layer->out->tensor->dim[2] = cl->filter_mult;
	}

	// bufferA size: (1D shape)
	// 2*ch_im_in*dim_kernel*dim_kernel
	layer->comp->shape = shape(2 * 2 * layer->in->tensor->dim[2] * cl->kernel.w * cl->kernel.h, 1, 1);
	// computational cost: K x K x Cin x Hour x Wout x Cout
	layer->stat.macc = cl->kernel.w * cl->kernel.h * layer->in->tensor->dim[2] * tensor_size(layer->out->tensor);
	return NN_SUCCESS;
}


nnom_status_t conv2d_run(nnom_layer_t *layer)
{
	nnom_conv2d_layer_t *cl = (nnom_conv2d_layer_t *)layer;

#ifdef NNOM_USING_CHW
	// CHW format
	local_convolve_CHW_q7_nonsquare(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weights->p_value, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_value, cl->bias_shift, cl->output_shift,
				layer->out->tensor->p_data,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0], (q15_t *)(layer->comp->mem->blk), NULL);
	return NN_SUCCESS;
#else
	// HWC format
	#ifdef NNOM_USING_CMSIS_NN
	//RGB
	// ch_im_in = 3, w = h
	if (layer->in->tensor->dim[2] == 3 && layer->in->tensor->dim[0] == layer->in->tensor->dim[1])
		return (nnom_status_t)arm_convolve_HWC_q7_RGB(
			layer->in->tensor->p_data, layer->in->tensor->dim[1], layer->in->tensor->dim[2],
			cl->weights->p_value,
			layer->out->tensor->dim[2],
			cl->kernel.w, cl->pad.w, cl->stride.w,
			cl->bias->p_value, cl->bias_shift,
			cl->output_shift, layer->out->tensor->p_data, layer->out->tensor->dim[1],
			(q15_t *)(layer->comp->mem->blk), NULL);

	// check if can use optimized function
	//	ch_im_in is multiple of 4
	//	ch_im_out is multiple of 2
	if (layer->in->tensor->dim[2] % 4 == 0 &&
		layer->out->tensor->dim[2] % 2 == 0)
	{
		// 1x1 fast
		if (cl->kernel.w == 1 && cl->kernel.h == 1)
			return (nnom_status_t)arm_convolve_1x1_HWC_q7_fast_nonsquare(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weights->p_value,
				layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_value, cl->bias_shift,
				cl->output_shift, layer->out->tensor->p_data, layer->out->tensor->dim[1], layer->out->tensor->dim[0],
				(q15_t *)(layer->comp->mem->blk), NULL);
		// opt square shape
		if (layer->in->tensor->dim[0] == layer->in->tensor->dim[1])
			return (nnom_status_t)arm_convolve_HWC_q7_fast(
				layer->in->tensor->p_data, layer->in->tensor->dim[1], layer->in->tensor->dim[2],
				cl->weights->p_value,
				layer->out->tensor->dim[2], cl->kernel.w, cl->pad.w, cl->stride.w,
				cl->bias->p_value, cl->bias_shift,
				cl->output_shift, layer->out->tensor->p_data,
				layer->out->tensor->dim[1], (q15_t *)(layer->comp->mem->blk), NULL);
		// opt none square shape
		else
			return (nnom_status_t)arm_convolve_HWC_q7_fast_nonsquare(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weights->p_value, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_value, cl->bias_shift, cl->output_shift,
				layer->out->tensor->p_data,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0], (q15_t *)(layer->comp->mem->blk), NULL);
	}
	// none optimized
	else
	{
		// none opt square shape
		if (layer->in->tensor->dim[0] == layer->in->tensor->dim[1])
			return (nnom_status_t)arm_convolve_HWC_q7_basic(
				layer->in->tensor->p_data, layer->in->tensor->dim[1], layer->in->tensor->dim[2],
				cl->weights->p_value,
				layer->out->tensor->dim[2], cl->kernel.w, cl->pad.w, cl->stride.w,
				cl->bias->p_value, cl->bias_shift,
				cl->output_shift, layer->out->tensor->p_data,
				layer->out->tensor->dim[1], (q15_t *)(layer->comp->mem->blk), NULL);
		// none opt none square shape
		else
			return (nnom_status_t)arm_convolve_HWC_q7_basic_nonsquare(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weights->p_value, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_value, cl->bias_shift, cl->output_shift,
				layer->out->tensor->p_data,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0], (q15_t *)(layer->comp->mem->blk), NULL);
	}
	// end of cmsis nn
	#else
	// local implementation
	local_convolve_HWC_q7_nonsquare(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weights->p_value, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_value, cl->bias_shift, cl->output_shift,
				layer->out->tensor->p_data,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0], (q15_t *)(layer->comp->mem->blk), NULL);
	return NN_SUCCESS;
	#endif
#endif // end of CHW/HWC
}

