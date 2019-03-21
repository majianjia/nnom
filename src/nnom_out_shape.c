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

#define NN_CEILIF(x,y) ((x+y-1)/y)

// this is call while output shape is not defined.
// this will set the output shape same as input shape, and it set only the primary IO
// this cannot be used as first layer, of course...
nnom_status_t default_out_shape(nnom_layer_t *layer)
{
	// get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;
	// output shape
	layer->out->shape = layer->in->shape;

	return NN_SUCCESS;
}

nnom_status_t input_out_shape(nnom_layer_t *layer)
{
	nnom_io_layer_t *cl = (nnom_io_layer_t *)layer;

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

	// output shape
	layer->in->mem->blk = cl->buf;

	layer->in->shape = cl->shape;
	layer->out->shape = cl->shape;

	return NN_SUCCESS;
}
nnom_status_t flatten_out_shape(nnom_layer_t *layer)
{ // get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;

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

	// output shape
	if (cl->padding_type == PADDING_SAME)
	{
		out->shape.h = NN_CEILIF(in->shape.h ,cl->stride.h);
		out->shape.w = NN_CEILIF(in->shape.w ,cl->stride.w);
		out->shape.c = cl->filter_mult; // output filter num
	}
	// new_height = new_width = (W-F+1)/S, round up
	else
	{
		out->shape.h = NN_CEILIF((in->shape.h - cl->kernel.h + 1) ,(cl->stride.h));
		out->shape.w = NN_CEILIF((in->shape.w - cl->kernel.w + 1) ,(cl->stride.w));
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

	if (cl->padding_type == PADDING_SAME)
	{
		out->shape.h = NN_CEILIF(in->shape.h ,cl->stride.h);
		out->shape.w = NN_CEILIF(in->shape.w ,cl->stride.w);
		out->shape.c = in->shape.c * cl->filter_mult; // for dw, is the multiplier for input channels
	}
	// new_height = new_width = (W-F+1)/S, round up
	else
	{
		out->shape.h = NN_CEILIF((in->shape.h - cl->kernel.h + 1) ,(cl->stride.h));
		out->shape.w = NN_CEILIF((in->shape.w - cl->kernel.w + 1) ,(cl->stride.w));
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

// the state buffer and computational buffer shape of the cell
nnom_status_t simplecell_out_shape(nnom_layer_t* layer, nnom_rnn_cell_t* cell)
{

}

// TODO
nnom_status_t rnn_out_shape(nnom_layer_t *layer)
{
	nnom_rnn_layer_t *cl = (nnom_rnn_layer_t *)layer;

	// get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;

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
		layer->out->shape.c = cl->cell->units;	 // output unit
	}
	else
	{
		layer->out->shape.h = 1;			  // batch?
		layer->out->shape.w = 1;			  // timestamp
		layer->out->shape.c = cl->cell->units; // output unit
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

	if (cl->padding_type == PADDING_SAME)
	{
		out->shape.h = NN_CEILIF((in->shape.h) ,(cl->stride.h));
		out->shape.w = NN_CEILIF((in->shape.w) ,(cl->stride.w));
		out->shape.c = in->shape.c;
	}
	else
	{
		out->shape.h = NN_CEILIF((in->shape.h - cl->kernel.h + 1) ,(cl->stride.h));
		out->shape.w = NN_CEILIF((in->shape.w - cl->kernel.w + 1) ,(cl->stride.w));
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

nnom_status_t sumpooling_out_shape(nnom_layer_t *layer)
{
	// avg pooling share the same output shape, stride, padding setting.
	maxpooling_out_shape(layer);

	// however, avg pooling require a computational buffer.
	layer->comp->shape = shape(4 * layer->out->shape.h * layer->out->shape.w * layer->out->shape.c, 1, 1);

	return NN_SUCCESS;
}


nnom_status_t upsample_out_shape(nnom_layer_t *layer)
{
	nnom_upsample_layer_t *cl = (nnom_upsample_layer_t *)layer;
	layer->in->shape = layer->in->hook.io->shape;

	layer->out->shape.c = layer->in->shape.c;
	layer->out->shape.h = layer->in->shape.h * cl->kernel.h;
	layer->out->shape.w = layer->in->shape.w * cl->kernel.w;

	return NN_SUCCESS;
}



nnom_status_t global_pooling_out_shape(nnom_layer_t *layer)
{
	nnom_maxpool_layer_t *cl = (nnom_maxpool_layer_t *)layer;
	nnom_layer_io_t *in = layer->in;
	nnom_layer_io_t *out = layer->out;
	//get the last layer's output as input shape
	in->shape = in->hook.io->shape;

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
	if (layer->type == NNOM_AVGPOOL || layer->type == NNOM_GLOBAL_AVGPOOL)
		layer->comp->shape = shape(2 * 1 * layer->in->shape.c, 1, 1);
	
	// additionally sumpool
	if (layer->type == NNOM_SUMPOOL || layer->type == NNOM_GLOBAL_SUMPOOL)
		layer->comp->shape = shape(4 * layer->out->shape.h * layer->out->shape.w * layer->out->shape.c, 1, 1);

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
			nnom_shape_data_t *out_axis = (nnom_shape_data_t *)((unsigned long)(&layer->out->shape) + i);
			*out_axis = 0;

			in = layer->in;
			while (in != NULL)
			{
				*out_axis += *(nnom_shape_data_t *)((unsigned long)(&in->shape) + i);
				in = in->aux;
			}
			continue;
		}

		// check others, all other must be same shape
		in = layer->in;
		while (in != NULL && in->aux != NULL)
		{
			if (*(nnom_shape_data_t *)((unsigned long)(&in->shape) + i) !=
				*(nnom_shape_data_t *)((unsigned long)(&in->aux->shape) + i))
				return NN_ARGUMENT_ERROR;
			in = in->aux;
		}

		// now set other axis
		*(nnom_shape_data_t *)((unsigned long)(&layer->out->shape) + i) =
			*(nnom_shape_data_t *)((unsigned long)(&layer->in->shape) + i);
	}

	return NN_SUCCESS;
}

// the shape of mutiple inputs are same as output
nnom_status_t same_io_shape_base_layer_out_shape(nnom_layer_t *layer)
{
	//get the last layer's output as input shape
	layer->in->shape = layer->in->hook.io->shape;

	layer->out->shape.h = layer->in->shape.h;
	layer->out->shape.w = layer->in->shape.w;
	layer->out->shape.c = layer->in->shape.c;

	return NN_SUCCESS;
}
