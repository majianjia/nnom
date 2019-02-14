/*
 * Copyright (c) 2018-2019
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: LGPL-3.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-02-05     Jianjia Ma   The first version
 */

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "nnom.h"
#include "nnom_layers.h"
#include "nnom_run.h"
#include "nnom_local.h"

#include "arm_math.h"

nnom_status_t input_run(nnom_layer_t *layer)
{
	nnom_io_layer_t *cl = (nnom_io_layer_t *)layer;
	memcpy(layer->in->mem->blk, cl->buf, shape_size(&layer->in->shape));
	return NN_SUCCESS;
}
nnom_status_t output_run(nnom_layer_t *layer)
{
	nnom_io_layer_t *cl = (nnom_io_layer_t *)layer;
	memcpy(cl->buf, layer->in->mem->blk, shape_size(&layer->in->shape)); // in->memory -> user memory
	return NN_SUCCESS;
}
nnom_status_t flatten_run(nnom_layer_t *layer)
{
	// you must be kidding me
	return NN_SUCCESS;
}

nnom_status_t dw_conv2d_run(nnom_layer_t *layer)
{
	nnom_status_t result;
	nnom_conv2d_layer_t *cl = (nnom_conv2d_layer_t *)layer;

	// CMSIS-NN only support 1 mulplipier in depthwise conv
	if (cl->filter_mult != 1 || layer->in->shape.c % 2 != 0 || layer->out->shape.c % 2)
		return NN_ARGUMENT_ERROR;

	// cmsis-nn dw does not support multiplier, we need to do it by our own
	result = (nnom_status_t)arm_depthwise_separable_conv_HWC_q7_nonsquare(
		layer->in->mem->blk,
		layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
		cl->weights->p_value,
		layer->in->shape.c,
		cl->kernel.w, cl->kernel.h,
		cl->pad.w, cl->pad.h,
		cl->stride.w, cl->stride.h,
		cl->bias->p_value,
		cl->bias_shift, cl->output_shift,
		layer->out->mem->blk,
		layer->out->shape.w, layer->out->shape.h, (q15_t *)(layer->comp->mem->blk), NULL);

	return result;
}

nnom_status_t conv2d_run(nnom_layer_t *layer)
{
	nnom_conv2d_layer_t *cl = (nnom_conv2d_layer_t *)layer;

	//RGB
	// ch_im_in = 3, w = h
	if (layer->in->shape.c == 3 && layer->in->shape.h == layer->in->shape.w)
		return (nnom_status_t)arm_convolve_HWC_q7_RGB(
			layer->in->mem->blk, layer->in->shape.w, layer->in->shape.c,
			cl->weights->p_value,
			layer->out->shape.c,
			cl->kernel.w, cl->pad.w, cl->stride.w,
			cl->bias->p_value, cl->bias_shift,
			cl->output_shift, layer->out->mem->blk, layer->out->shape.w,
			(q15_t *)(layer->comp->mem->blk), NULL);

	// check if can use optimized function
	//	ch_im_in is multiple of 4
	//	ch_im_out is multiple of 2
	if (layer->in->shape.c % 4 == 0 &&
		layer->out->shape.c % 2 == 0)
	{
		// 1x1 fast
		if (cl->kernel.w == 1 && cl->kernel.h == 1)
			return (nnom_status_t)arm_convolve_1x1_HWC_q7_fast_nonsquare(
				layer->in->mem->blk,
				layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
				cl->weights->p_value,
				layer->out->shape.c,
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_value, cl->bias_shift,
				cl->output_shift, layer->out->mem->blk, layer->out->shape.w, layer->out->shape.h,
				(q15_t *)(layer->comp->mem->blk), NULL);

		// opt square shape
		if (layer->in->shape.h == layer->in->shape.w)
			return (nnom_status_t)arm_convolve_HWC_q7_fast(
				layer->in->mem->blk, layer->in->shape.w, layer->in->shape.c,
				cl->weights->p_value,
				layer->out->shape.c, cl->kernel.w, cl->pad.w, cl->stride.w,
				cl->bias->p_value, cl->bias_shift,
				cl->output_shift, layer->out->mem->blk,
				layer->out->shape.w, (q15_t *)(layer->comp->mem->blk), NULL);
		// opt none square shape
		else
			return (nnom_status_t)arm_convolve_HWC_q7_fast_nonsquare(
				layer->in->mem->blk,
				layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
				cl->weights->p_value, layer->out->shape.c,
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_value, cl->bias_shift, cl->output_shift,
				layer->out->mem->blk,
				layer->out->shape.w, layer->out->shape.h, (q15_t *)(layer->comp->mem->blk), NULL);
	}
	// none optimized
	else
	{
		// none opt square shape
		if (layer->in->shape.h == layer->in->shape.w)
			return (nnom_status_t)arm_convolve_HWC_q7_basic(
				layer->in->mem->blk, layer->in->shape.w, layer->in->shape.c,
				cl->weights->p_value,
				layer->out->shape.c, cl->kernel.w, cl->pad.w, cl->stride.w,
				cl->bias->p_value, cl->bias_shift,
				cl->output_shift, layer->out->mem->blk,
				layer->out->shape.w, (q15_t *)(layer->comp->mem->blk), NULL);
		// none opt none square shape
		else
			return (nnom_status_t)arm_convolve_HWC_q7_basic_nonsquare(
				layer->in->mem->blk,
				layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
				cl->weights->p_value, layer->out->shape.c,
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_value, cl->bias_shift, cl->output_shift,
				layer->out->mem->blk,
				layer->out->shape.w, layer->out->shape.h, (q15_t *)(layer->comp->mem->blk), NULL);
	}
}

nnom_status_t cell_simple_rnn_run(nnom_layer_t *layer)
{

	return NN_SUCCESS;
}

nnom_status_t rnn_run(nnom_layer_t *layer)
{
	nnom_status_t result;
	nnom_rnn_layer_t *cl = (nnom_rnn_layer_t *)(layer);

	for (uint32_t round = 0; round < cl->cell->unit; round++)
	{
		result = cl->cell->run(layer);
	}
	return result;
}

nnom_status_t dense_run(nnom_layer_t *layer)
{
	nnom_status_t result;
	nnom_dense_layer_t *cl = (nnom_dense_layer_t *)(layer);

	// test, optimize
#if !(DENSE_WEIGHT_OPT)
	result = (nnom_status_t)arm_fully_connected_q7(
		layer->in->mem->blk,
		cl->weights->p_value,
		layer->in->shape.h, layer->out->shape.h,
		cl->bias_shift, cl->output_shift,
		cl->bias->p_value,
		layer->out->mem->blk, (q15_t *)(layer->comp->mem->blk));
#else
	result = (nnom_status_t)arm_fully_connected_q7_opt(
		layer->in->mem->blk,
		cl->weights->p_value,
		layer->in->shape.h, layer->out->shape.h,
		cl->bias_shift, cl->output_shift,
		cl->bias->p_value,
		layer->out->mem->blk, (q15_t *)(layer->comp->mem->blk));
#endif

	return result;
}

nnom_status_t activation_run(nnom_layer_t *layer)
{
	nnom_activation_layer_t *cl = (nnom_activation_layer_t *)layer;
	// set up buf
	cl->act->data = layer->in->mem->blk;
	cl->act->size = layer->out->shape.h * layer->out->shape.w * layer->out->shape.c;
	cl->act->fmt = layer->in->qfmt;
	return cl->act->run(layer, cl->act);
}

nnom_status_t relu_run(nnom_layer_t *layer)
{
	arm_relu_q7(layer->in->mem->blk, layer->out->shape.h * layer->out->shape.w * layer->out->shape.c);

	return NN_SUCCESS;
}
nnom_status_t tanh_run(nnom_layer_t *layer)
{
	arm_nn_activations_direct_q7(layer->in->mem->blk,
								 layer->out->shape.h * layer->out->shape.w * layer->out->shape.c,
								 layer->in->qfmt.n,
								 ARM_TANH);

	return NN_SUCCESS;
}
nnom_status_t sigmoid_run(nnom_layer_t *layer)
{
	arm_nn_activations_direct_q7(layer->in->mem->blk,
								 layer->out->shape.h * layer->out->shape.w * layer->out->shape.c,
								 layer->in->qfmt.n,
								 ARM_SIGMOID);
	return NN_SUCCESS;
}

nnom_status_t maxpool_run(nnom_layer_t *layer)
{
	nnom_maxpool_layer_t *cl = (nnom_maxpool_layer_t *)(layer);

	// 1D
	if (layer->in->shape.h == 1)
	{
		arm_maxpool_1d_q7_HWC(
			layer->in->mem->blk,
			layer->in->shape.w, layer->in->shape.c,
			cl->kernel.w, cl->pad.w,
			cl->stride.w,
			layer->out->shape.w,
			NULL,
			layer->out->mem->blk);
	}
	// 2D, square
	else if (layer->in->shape.w == layer->in->shape.h)
	{
		arm_maxpool_q7_HWC(
			layer->in->mem->blk,
			layer->in->shape.w, layer->in->shape.c,
			cl->kernel.w, cl->pad.w,
			cl->stride.w,
			layer->out->shape.w,
			NULL,
			layer->out->mem->blk);
	}
	else 
		return NN_ARGUMENT_ERROR;

	return NN_SUCCESS;
}

nnom_status_t avgpool_run(nnom_layer_t *layer)
{
	nnom_avgpool_layer_t *cl = (nnom_avgpool_layer_t *)(layer);

	// 1D is not working yet.
	if (layer->in->shape.h == 1)
	{
		arm_avepool_1d_q7_HWC(
			layer->in->mem->blk,
			layer->in->shape.w, layer->in->shape.c,
			cl->kernel.w, cl->pad.w,
			cl->stride.w,
			layer->out->shape.w,
			layer->comp->mem->blk,
			layer->out->mem->blk);
	}
	// 2D, square
	else if (layer->in->shape.w == layer->in->shape.h)
	{
		arm_avepool_q7_HWC(
			layer->in->mem->blk,
			layer->in->shape.w, layer->in->shape.c,
			cl->kernel.w, cl->pad.w,
			cl->stride.w,
			layer->out->shape.w,
			layer->comp->mem->blk,
			layer->out->mem->blk);
	}
	else
		return NN_ARGUMENT_ERROR;

	return NN_SUCCESS;
}

nnom_status_t softmax_run(nnom_layer_t *layer)
{
	arm_softmax_q7(layer->in->mem->blk, layer->out->shape.h, layer->out->mem->blk);
	return NN_SUCCESS;
}

nnom_status_t concat_run(nnom_layer_t *layer)
{
	// by default, concat layer has mutiple (>=2) input and 1 output.
	nnom_concat_layer_t *cl = (nnom_concat_layer_t *)layer;
	uint32_t shape_element_num = sizeof(nnom_shape_t) / sizeof(nnom_shape_data_t);
	size_t width = sizeof(nnom_shape_data_t);
	nnom_shape_axis_t *out_shape = (nnom_shape_axis_t *)(&layer->out->shape); // get the shape.axis[0,1,2...] access to shape type
	uint32_t offset;
	nnom_layer_io_t *in, *out;

	in = layer->in;
	out = layer->out;

	// last axis, shape c
	if (cl->axis < 0)
		offset = (shape_element_num + cl->axis);
	else
		offset = cl->axis;

	// concat by different axis, TODO, change to nested loop
	// the concat axis might be different, means that, the block size for each input could be different
	if (offset == 0)
	{
		uint8_t *pin;
		uint8_t *pout = out->mem->blk;
		in = layer->in;
		while (in != NULL)
		{
			pin = in->mem->blk;
			memcpy(pout, pin, shape_size(&in->shape));
			pout += shape_size(&in->shape);

			in = in->aux;
		}
	}
	else if (offset == 1)
	{
		uint8_t *pin;
		uint8_t *pout = out->mem->blk;
		uint32_t block_size;

		for (int j = 0; j < out_shape->axis[0]; j++)
		{
			in = layer->in;
			while (in != NULL)
			{
				block_size = in->shape.c * in->shape.w;
				pin = (uint8_t *)in->mem->blk + j * block_size;
				memcpy(pout, pin, block_size);
				pout += block_size;

				in = in->aux;
			}
		}
	}
	else if (offset == 2)
	{
		uint8_t *pin;
		uint8_t *pout = out->mem->blk;
		uint32_t block_size;

		for (int j = 0; j < out_shape->axis[1] * out_shape->axis[0]; j++)
		{
			in = layer->in;
			while (in != NULL)
			{
				block_size = in->shape.c;
				pin = (uint8_t *)in->mem->blk + j * block_size;
				memcpy(pout, pin, block_size);
				pout += block_size;

				in = in->aux;
			}
		}
	}

	return NN_SUCCESS;
}

nnom_status_t concat_run_bk(nnom_layer_t *layer)
{
	// by default, concat layer has 2 input and 1 output.
	nnom_concat_layer_t *cl = (nnom_concat_layer_t *)layer;
	uint32_t shape_element_num = sizeof(nnom_shape_t) / sizeof(nnom_shape_data_t);
	size_t width = sizeof(nnom_shape_data_t);
	nnom_shape_axis_t *out_shape = (nnom_shape_axis_t *)(&layer->out->shape); // get the shape.axis[0,1,2...] access to shape type
	uint32_t offset;
	nnom_layer_io_t *in1, *in2, *out;

	in1 = layer->in;
	in2 = layer->in->aux;
	out = layer->out;

	// last axis, shape c
	if (cl->axis < 0)
		offset = (shape_element_num + cl->axis);
	else
		offset = cl->axis;

	// concat by different axis, TODO, change to nested for loop
	if (offset == 0)
	{
		memcpy(out->mem->blk, in1->mem->blk, shape_size(&in1->shape));
		memcpy((void *)((uint32_t)out->mem->blk + shape_size(&in1->shape)),
			   in2->mem->blk, shape_size(&in2->shape));
	}
	else if (offset == 1)
	{
		uint8_t *pout = out->mem->blk;
		uint8_t *pin1 = in1->mem->blk;
		uint8_t *pin2 = in2->mem->blk;

		for (int j = 0; j < out_shape->axis[0]; j++)
		{
			memcpy(pout, pin1, in1->shape.c * in1->shape.w);
			memcpy(pout + in1->shape.c * in1->shape.w, pin2, in2->shape.c * in2->shape.w);
			pin1 += in1->shape.c * in1->shape.w;
			pin2 += in2->shape.c * in2->shape.w;
			pout += in1->shape.c * in1->shape.w + in2->shape.c * in2->shape.w;
		}
	}
	else if (offset == 2)
	{
		uint8_t *pout = out->mem->blk;
		uint8_t *pin1 = in1->mem->blk;
		uint8_t *pin2 = in2->mem->blk;

		for (int j = 0; j < out_shape->axis[1] * out_shape->axis[0]; j++)
		{
			memcpy(pout, pin1, in1->shape.c);
			memcpy(pout + in1->shape.c, pin2, in2->shape.c);
			pin1 += in1->shape.c;
			pin2 += in2->shape.c;
			pout += in1->shape.c + in2->shape.c;
		}
	}

	return NN_SUCCESS;
}

nnom_status_t add_run(nnom_layer_t *layer)
{
	nnom_layer_io_t *in;
	size_t size = shape_size(&layer->in->shape);

	// adding the first 2 matrix
	arm_add_q7(layer->in->mem->blk,
			   layer->in->aux->mem->blk,
			   layer->out->mem->blk,
			   size);

	// if there is 3rd or more
	if (layer->in->aux->aux != NULL)
	{
		in = layer->in->aux->aux;
		while (in != NULL)
		{
			arm_add_q7(in->mem->blk,
					   layer->out->mem->blk,
					   layer->out->mem->blk,
					   size);

			in = in->aux;
		}
	}

	return NN_SUCCESS;
}

nnom_status_t sub_run(nnom_layer_t *layer)
{
	nnom_layer_io_t *in;
	size_t size = shape_size(&layer->in->shape);

	// adding the first 2 matrix
	arm_sub_q7(layer->in->mem->blk,
			   layer->in->aux->mem->blk,
			   layer->out->mem->blk,
			   size);

	// if there is 3rd or more
	if (layer->in->aux->aux != NULL)
	{
		in = layer->in->aux->aux;
		while (in != NULL)
		{
			arm_sub_q7(in->mem->blk,
					   layer->out->mem->blk,
					   layer->out->mem->blk,
					   size);

			in = in->aux;
		}
	}
	return NN_SUCCESS;
}

nnom_status_t mult_run(nnom_layer_t *layer)
{
	nnom_layer_io_t *in;
	size_t size = shape_size(&layer->in->shape);

	// adding the first 2 matrix
	arm_mult_q7(layer->in->mem->blk,
				layer->in->aux->mem->blk,
				layer->out->mem->blk,
				size);

	// if there is 3rd or more
	if (layer->in->aux->aux != NULL)
	{
		in = layer->in->aux->aux;
		while (in != NULL)
		{
			arm_mult_q7(in->mem->blk,
						layer->out->mem->blk,
						layer->out->mem->blk,
						size);

			in = in->aux;
		}
	}
	return NN_SUCCESS;
}
