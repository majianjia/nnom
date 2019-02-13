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
#include "nnom_out_shape.h"
#include "nnom_run.h"
#include "math.h"

// this is call while output shape is not defined.
// this will set the output shape same as input shape, and it set only the primary IO
// this cannot be used as first layer, of course...
nnom_status_t default_out_shape(nnom_layer_t *layer)
{
	// get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;
	// test,
	layer->in->qfmt = layer->in->hook.io->qfmt;
	layer->out->qfmt = layer->in->qfmt;
	// output shape
	layer->out->shape = layer->in->shape;

	return NN_SUCCESS;
}

nnom_status_t input_out_shape(nnom_layer_t *layer)
{
	nnom_io_layer_t *cl = (nnom_io_layer_t *)layer;

	// the input layer's input shape will be set previously, we dont manually set it.
	layer->in->qfmt = cl->format;
	layer->out->qfmt = layer->in->qfmt;

	// output shape
	layer->in->mem->blk = cl->buf;
	layer->in->shape = cl->shape;
	layer->out->shape = cl->shape;

	return NN_SUCCESS;
}
nnom_status_t output_out_shape(nnom_layer_t *layer)
{
	nnom_io_layer_t *cl = (nnom_io_layer_t *)layer;

	// get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;

	// test,
	layer->in->qfmt = layer->in->hook.io->qfmt;
	layer->out->qfmt = layer->in->qfmt;

	// output shape
	layer->in->mem->blk = cl->buf;

	layer->in->shape = cl->shape;
	layer->out->shape = cl->shape;

	return NN_SUCCESS;
}
nnom_status_t flatten_out_shape(nnom_layer_t *layer)
{ // get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;

	// test,
	layer->in->qfmt = layer->in->hook.io->qfmt;
	layer->out->qfmt = layer->in->qfmt;

	// output shape
	layer->out->shape.h = layer->in->shape.h * layer->in->shape.w * layer->in->shape.c;
	layer->out->shape.w = 1;
	layer->out->shape.c = 1;

	return NN_SUCCESS;
}

nnom_status_t conv2d_out_shape(nnom_layer_t *layer)
{
	nnom_conv2d_layer_t *cl = (nnom_conv2d_layer_t *)layer;
	nnom_layer_io_t *in = layer->in;
	nnom_layer_io_t *out = layer->out;

	// get the last layer's output as input shape
	in->shape = in->hook.io->shape;

	// test,
	layer->in->qfmt = layer->in->hook.io->qfmt;
	// test, output fmt and out shift
	layer->out->qfmt = layer->in->qfmt;

	// output shape
	if (cl->padding_type == PADDING_SAME)
	{
		out->shape.h = ceilf((float)in->shape.h / (float)cl->stride.h);
		out->shape.w = ceilf((float)in->shape.w / (float)cl->stride.w);
		out->shape.c = cl->filter_mult; // output filter num
	}
	// new_height = new_width = (W-F+1)/S, round up
	else
	{
		out->shape.h = ceilf((float)(in->shape.h - cl->kernel.h + 1) / (float)(cl->stride.h));
		out->shape.w = ceilf((float)(in->shape.w - cl->kernel.w + 1) / (float)(cl->stride.w));
		out->shape.c = cl->filter_mult;
	}
	// bufferA size: (1D shape)
	layer->comp->shape = shape(1, 2 * 2 * layer->out->shape.c * cl->kernel.w * cl->kernel.h, 1);
	// computational cost: K x K x Cin x Hour x Wout x Cout
	layer->stat.macc = cl->kernel.w * cl->kernel.h * in->shape.c * out->shape.w * out->shape.h * out->shape.c;
	return NN_SUCCESS;
}
nnom_status_t dw_conv2d_out_shape(nnom_layer_t *layer)
{
	nnom_conv2d_layer_t *cl = (nnom_conv2d_layer_t *)layer;
	nnom_layer_io_t *in = layer->in;
	nnom_layer_io_t *out = layer->out;

	//get the last layer's output as input shape
	in->shape = in->hook.io->shape;

	// test,
	layer->in->qfmt = layer->in->hook.io->qfmt;
	// test, output fmt and out shift
	layer->out->qfmt = layer->in->qfmt;

	if (cl->padding_type == PADDING_SAME)
	{
		out->shape.h = ceilf((float)in->shape.h / (float)cl->stride.h);
		out->shape.w = ceilf((float)in->shape.w / (float)cl->stride.w);
		out->shape.c = in->shape.c * cl->filter_mult; // for dw, is the multiplier for input channels
	}
	// new_height = new_width = (W-F+1)/S, round up
	else
	{
		out->shape.h = ceilf((float)(in->shape.h - cl->kernel.h + 1) / (float)(cl->stride.h));
		out->shape.w = ceilf((float)(in->shape.w - cl->kernel.w + 1) / (float)(cl->stride.w));
		out->shape.c = in->shape.c * cl->filter_mult;
	}
	// bufferA size: (1D shape)
	layer->comp->shape = shape(1, 2 * 2 * (layer->out->shape.c / cl->filter_mult) * cl->kernel.w * cl->kernel.h, 1);

	// computational cost: K x K x Cin x Hour x Wout x Multiplier
	layer->stat.macc = cl->kernel.w * cl->kernel.h * in->shape.c * out->shape.w * out->shape.h * cl->filter_mult;
	return NN_SUCCESS;
}

nnom_status_t dense_out_shape(nnom_layer_t *layer)
{
	nnom_dense_layer_t *cl = (nnom_dense_layer_t *)layer;
	nnom_layer_io_t *in = layer->in;
	nnom_layer_io_t *out = layer->out;

	//get the last layer's output as input shape
	in->shape = in->hook.io->shape;

	// test,
	layer->in->qfmt = layer->in->hook.io->qfmt;
	// test, output fmt and out shift
	layer->out->qfmt = layer->in->qfmt;
	//cl->output_shift =
	//cl->bias_shift =

	// incase the input hasnt been flattened.
	in->shape.h = in->shape.h * in->shape.w * in->shape.c;
	in->shape.w = 1;
	in->shape.c = 1;

	out->shape.h = cl->output_unit;
	out->shape.w = 1;
	out->shape.c = 1;

	// vec_buffer size: dim_vec
	layer->comp->shape = shape(1, in->shape.h, 1);

	// computational cost: In * out
	layer->stat.macc = in->shape.h * out->shape.h;
	return NN_SUCCESS;
}

// TODO
nnom_status_t rnn_out_shape(nnom_layer_t *layer)
{
	nnom_rnn_layer_t *cl = (nnom_rnn_layer_t *)layer;

	// get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;
	// test,
	layer->in->qfmt = layer->in->hook.io->qfmt;
	layer->out->qfmt = layer->in->qfmt;
	// output shape
	//layer->out->shape = layer->in->shape;

	// TODO
	// calculate computational and stat buf size
	layer->comp = 0;

	// calculate output shape according to the return_sequence
	if (cl->return_sequence)
	{
		layer->out->shape.h = 1;				  // batch?
		layer->out->shape.w = layer->in->shape.w; // timestamp (same timestamps)
		layer->out->shape.c = cl->cell->unit;	 // output unit
	}
	else
	{
		layer->out->shape.h = 1;			  // batch?
		layer->out->shape.w = 1;			  // timestamp
		layer->out->shape.c = cl->cell->unit; // output unit
	}
	return NN_SUCCESS;
}

nnom_status_t maxpooling_out_shape(nnom_layer_t *layer)
{
	nnom_maxpool_layer_t *cl = (nnom_maxpool_layer_t *)layer;
	nnom_layer_io_t *in = layer->in;
	nnom_layer_io_t *out = layer->out;
	//get the last layer's output as input shape
	in->shape = in->hook.io->shape;

	// test, output fmt and out shift
	layer->in->qfmt = layer->in->hook.io->qfmt;
	layer->out->qfmt = layer->in->qfmt;

	if (cl->padding_type == PADDING_SAME)
	{
		out->shape.h = ceilf((float)(in->shape.h) / (float)(cl->stride.h));
		out->shape.w = ceilf((float)(in->shape.w) / (float)(cl->stride.w));
		out->shape.c = in->shape.c;
	}
	else
	{
		out->shape.h = ceilf((float)(in->shape.h - cl->kernel.h + 1) / (float)(cl->stride.h));
		out->shape.w = ceilf((float)(in->shape.w - cl->kernel.w + 1) / (float)(cl->stride.w));
		out->shape.c = in->shape.c;
	}

	return NN_SUCCESS;
}

nnom_status_t avgpooling_out_shape(nnom_layer_t *layer)
{
	// avg pooling share the same output shape, stride, padding setting. 
	maxpooling_out_shape(layer);
	
	// however, avg pooling require a computational buffer. 
	layer->comp->shape = shape(2 * 1 * layer->in->shape.c, 1, 1);

	return NN_SUCCESS;
}

nnom_status_t global_pooling_out_shape(nnom_layer_t *layer)
{
	nnom_maxpool_layer_t *cl = (nnom_maxpool_layer_t *)layer;
	nnom_layer_io_t *in = layer->in;
	nnom_layer_io_t *out = layer->out;
	//get the last layer's output as input shape
	in->shape = in->hook.io->shape;

	// test, output fmt and out shift
	layer->in->qfmt = layer->in->hook.io->qfmt;
	layer->out->qfmt = layer->in->qfmt;
	
	// global pooling
	// output (h = 1, w = 1, same channels)
	out->shape.h = 1;
	out->shape.w = 1;
	out->shape.c = in->shape.c; 
	
	// different from other output_shape(), the kernel..padding left by layer API needs to be set in here
	// due to the *_run() methods of global pooling are using the normall pooling's. 
	// fill in the parameters left by layer APIs (GlobalAvgPool and MaxAvgPool) 
	cl->kernel = in->shape;
	cl->stride = shape(1, 1, 1);
	cl->pad = shape(0, 0, 0);
	cl->padding_type = PADDING_VALID;
	
	// additionally avg pooling require computational buffer, which is  2*dim_im_out*ch_im_in
	if(layer->type == NNOM_AVGPOOL || layer->type == NNOM_GLOBAL_AVGPOOL)
		layer->comp->shape = shape(2 * 1 * layer->in->shape.c, 1, 1);

	return NN_SUCCESS;
}


nnom_status_t concatenate_out_shape(nnom_layer_t *layer)
{
	nnom_concat_layer_t *cl = (nnom_concat_layer_t *)layer;
	nnom_layer_io_t *in;
	uint32_t in_num = 0;
	uint32_t offset;
	int32_t shape_element_num;

	// for each input module, copy the shape from the output of last layer
	in = layer->in;
	while (in != NULL)
	{
		//get the last layer's output as input shape
		in->shape = in->hook.io->shape;
		in = in->aux;
		in_num++;
	}

	// get how many element in shape
	shape_element_num = sizeof(nnom_shape_t) / sizeof(nnom_shape_data_t);
	if (cl->axis >= shape_element_num || cl->axis <= -shape_element_num)
		return NN_ARGUMENT_ERROR;

	// last axis, shape c
	if (cl->axis < 0)
		offset = shape_element_num + cl->axis;
	else
		offset = cl->axis;

	// do the work
	for (uint32_t i = 0; i < shape_element_num * sizeof(nnom_shape_data_t); i += sizeof(nnom_shape_data_t))
	{
		// exclue the concat axies
		if (i == offset * sizeof(nnom_shape_data_t))
		{
			nnom_shape_data_t *out_axis = (nnom_shape_data_t *)((uint32_t)(&layer->out->shape) + i);
			*out_axis = 0;

			in = layer->in;
			while (in != NULL)
			{
				*out_axis += *(nnom_shape_data_t *)((uint32_t)(&in->shape) + i);
				in = in->aux;
			}
			continue;
		}

		// check others, all other must be same shape
		in = layer->in;
		while (in != NULL && in->aux != NULL)
		{
			if (*(nnom_shape_data_t *)((uint32_t)(&in->shape) + i) !=
				*(nnom_shape_data_t *)((uint32_t)(&in->aux->shape) + i))
				return NN_ARGUMENT_ERROR;
			in = in->aux;
		}

		// now set other axis
		*(nnom_shape_data_t *)((uint32_t)(&layer->out->shape) + i) =
			*(nnom_shape_data_t *)((uint32_t)(&layer->in->shape) + i);
	}

	return NN_SUCCESS;
}

// deprecated.
nnom_status_t same_shape_2in_1out_out_shape(nnom_layer_t *layer)
{
	//get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;			// primary input
	layer->in->aux->shape = layer->in->aux->hook.io->shape; // aux input

	// test, output fmt and out shift
	layer->in->qfmt = layer->in->hook.io->qfmt;
	layer->in->aux->qfmt = layer->in->aux->hook.io->qfmt;
	// mutiple input, check if adjust Qnm is necessary.
	// layer->out->qfmt.m = layer->in->qfmt.m;
	// get the bigger one, the smaller one will then be shift while doing concatenation
	layer->out->qfmt.n = layer->in->qfmt.n < layer->in->aux->qfmt.n ? layer->in->qfmt.n : layer->in->aux->qfmt.n;

	if (layer->in->shape.h != layer->in->aux->shape.h ||
		layer->in->shape.w != layer->in->aux->shape.w ||
		layer->in->shape.c != layer->in->aux->shape.c)
		return NN_ARGUMENT_ERROR;

	layer->out->shape.h = layer->in->shape.h;
	layer->out->shape.w = layer->in->shape.w;
	layer->out->shape.c = layer->in->shape.c;

	return NN_SUCCESS;
}

// the shape of mutiple inputs are same as output
nnom_status_t same_io_shape_base_layer_out_shape(nnom_layer_t *layer)
{
	//get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;

	// test, output fmt and out shift
	layer->in->qfmt = layer->in->hook.io->qfmt;

	layer->out->shape.h = layer->in->shape.h;
	layer->out->shape.w = layer->in->shape.w;
	layer->out->shape.c = layer->in->shape.c;

	return NN_SUCCESS;
}
