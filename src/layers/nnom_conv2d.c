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
#include "layers/nnom_conv2d.h"


#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

nnom_status_t conv2d_run(nnom_layer_t *layer);
nnom_status_t conv2d_build(nnom_layer_t *layer);
nnom_status_t conv2d_free(nnom_layer_t *layer);

// a machine friendly api, with suffix _s for structured configuration.  
nnom_layer_t *conv2d_s(nnom_conv2d_config_t *config)
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

	// set buf state
	in->type = NNOM_TENSOR_BUF_TEMP;
	out->type = NNOM_TENSOR_BUF_TEMP;
	comp->type = NNOM_TENSOR_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	layer->super.comp = comp;
	// set run method & output shape
	layer->super.run = conv2d_run;
	layer->super.build = conv2d_build;
	layer->super.free = conv2d_free;

	// save the config
	layer->super.config = config;

	// get the private parameters
	layer->kernel = kernel(config->kernel_size[0], config->kernel_size[1]);
	layer->stride = stride(config->stride_size[0], config->stride_size[1]);
	layer->filter_mult = config->filter_size; // for convs, this means filter number
	layer->padding_type = config->padding;

	// get bias and weight tensor, this should be created by script. 
	layer->weight = config->weight;
	layer->bias = config->bias;

	// padding
	if (layer->padding_type == PADDING_SAME)
	{
		layer->pad.h = (config->padding_size[0]- 1) / 2;
		layer->pad.w = (config->padding_size[1] - 1) / 2;
		layer->pad.c = (1 - 1) / 2;
	}

	return (nnom_layer_t *)layer;
}


// Conv2D
// multiplier of (output/input channel),
// shape of kernal, shape of strides, weight struct, bias struct
nnom_layer_t *Conv2D(uint32_t filters, nnom_3d_shape_t k, nnom_3d_shape_t s, nnom_padding_t pad_type,
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
	in->type = NNOM_TENSOR_BUF_TEMP;
	out->type = NNOM_TENSOR_BUF_TEMP;
	comp->type = NNOM_TENSOR_BUF_TEMP;
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
	layer->dilation = stride(1, 1); 	//  the old api dosent support it. 
	layer->filter_mult = filters; 		// for convs, this means filter number
	layer->padding_type = pad_type;

	// create weight and bias tensor
	layer->weight = new_tensor(NNOM_QTYPE_PER_TENSOR, 4, filters);
	layer->bias = new_tensor(NNOM_QTYPE_PER_TENSOR, 1, filters);

	// configure weight tensor manually to support new tensor based backends. 
	// needs to be very careful
	{
		// config weight 
		nnom_shape_data_t dim[4] = {k.h, k.w, k.c, filters};
		*(layer->weight->q_offset) = 0;			// we have no support of offset here
		*(layer->weight->q_dec) = w->shift;		// this is not even correct
		layer->weight->p_data = (void*)w->p_value;
		layer->weight->bitwidth = 8;
		memcpy(layer->weight->dim, dim, layer->weight->num_dim * sizeof(nnom_shape_data_t));

		// config bias 
		dim[0] = filters;
		*(layer->bias->q_offset) = 0;			// we have no support of offset here
		*(layer->bias->q_dec) = b->shift;		// this is not even correct
		layer->bias->p_data = (void*) b->p_value;
		layer->bias->bitwidth = 8;
		memcpy(layer->bias->dim, dim, layer->bias->num_dim * sizeof(nnom_shape_data_t));
	}

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

	// create new tensor for the output
	layer->out->tensor = new_tensor(NNOM_QTYPE_PER_TENSOR, layer->in->tensor->num_dim, cl->filter_mult);
	// copy then change later. 
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);

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
	layer->comp->size = 2 * 2 * layer->in->tensor->dim[2] * cl->kernel.w * cl->kernel.h;
	// computational cost: K x K x Cin x Hour x Wout x Cout
	layer->stat.macc = cl->kernel.w * cl->kernel.h * layer->in->tensor->dim[2] * tensor_size(layer->out->tensor);
	return NN_SUCCESS;
}

nnom_status_t conv2d_free(nnom_layer_t *layer)
{
	// free weight and bias tensor when we are not initialised from structured configuration. 
	if(layer->config != NULL)
	{
		nnom_conv2d_layer_t* cl = (nnom_conv2d_layer_t*)layer;
		delete_tensor(cl->weight);
		delete_tensor(cl->bias);
	}

	return NN_SUCCESS;
}


nnom_status_t conv2d_run(nnom_layer_t *layer)
{
	nnom_conv2d_layer_t *cl = (nnom_conv2d_layer_t *)layer;
	int32_t bias_shift = cl->bias->q_dec[0];			// this is not correct but a temporary fix solution for backward compatibility.
	int32_t output_shift = cl->weight->q_dec[0];

#ifdef NNOM_USING_CHW
	// CHW format
	local_convolve_CHW_q7_nonsquare(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weight->p_data, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_data, bias_shift, output_shift,
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
			cl->weight->p_data,
			layer->out->tensor->dim[2],
			cl->kernel.w, cl->pad.w, cl->stride.w,
			cl->bias->p_data, bias_shift,
			output_shift, layer->out->tensor->p_data, layer->out->tensor->dim[1],
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
				cl->weight->p_data,
				layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_data, bias_shift,
				output_shift, layer->out->tensor->p_data, layer->out->tensor->dim[1], layer->out->tensor->dim[0],
				(q15_t *)(layer->comp->mem->blk), NULL);
		// opt square shape
		if (layer->in->tensor->dim[0] == layer->in->tensor->dim[1])
			return (nnom_status_t)arm_convolve_HWC_q7_fast(
				layer->in->tensor->p_data, layer->in->tensor->dim[1], layer->in->tensor->dim[2],
				cl->weight->p_data,
				layer->out->tensor->dim[2], cl->kernel.w, cl->pad.w, cl->stride.w,
				cl->bias->p_data, bias_shift,
				output_shift, layer->out->tensor->p_data,
				layer->out->tensor->dim[1], (q15_t *)(layer->comp->mem->blk), NULL);
		// opt none square shape
		else
			return (nnom_status_t)arm_convolve_HWC_q7_fast_nonsquare(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weight->p_data, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_data, bias_shift, output_shift,
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
				cl->weight->p_data,
				layer->out->tensor->dim[2], cl->kernel.w, cl->pad.w, cl->stride.w,
				cl->bias->p_data, bias_shift,
				output_shift, layer->out->tensor->p_data,
				layer->out->tensor->dim[1], (q15_t *)(layer->comp->mem->blk), NULL);
		// none opt none square shape
		else
			return (nnom_status_t)arm_convolve_HWC_q7_basic_nonsquare(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weight->p_data, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_data, bias_shift, output_shift,
				layer->out->tensor->p_data,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0], (q15_t *)(layer->comp->mem->blk), NULL);
	}
	// end of cmsis nn
	#else
	// local implementation
	local_convolve_HWC_q7_nonsquare(
				layer->in->tensor->p_data,
				layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
				cl->weight->p_data, layer->out->tensor->dim[2],
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_data, bias_shift, output_shift,
				layer->out->tensor->p_data,
				layer->out->tensor->dim[1], layer->out->tensor->dim[0], (q15_t *)(layer->comp->mem->blk), NULL);
	return NN_SUCCESS;
	#endif
#endif // end of CHW/HWC
}

