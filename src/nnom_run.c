/*
 * Copyright (c) 2018-2019
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
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

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

nnom_status_t input_run(nnom_layer_t *layer)
{
	nnom_io_layer_t *cl = (nnom_io_layer_t *)layer;
#ifdef NNOM_USING_CHW
	hwc2chw_q7(layer->in->shape, cl->buf, layer->in->mem->blk); // 
#else
	memcpy(layer->in->mem->blk, cl->buf, shape_size(&layer->in->shape));
#endif
	return NN_SUCCESS;
}
nnom_status_t output_run(nnom_layer_t *layer)
{
	nnom_io_layer_t *cl = (nnom_io_layer_t *)layer;
	memcpy(cl->buf, layer->in->mem->blk, shape_size(&layer->in->shape)); // in->memory -> user memory
	return NN_SUCCESS;
}

// change format from CHW to HWC
// the shape of the data, input data, output data
void hwc2chw_q7(nnom_shape_t shape, q7_t* p_in, q7_t* p_out)
{
	for(int c=0; c<shape.c; c++)
	{	
		for(int h=0; h<shape.h; h++)
		{
			for(int w=0; w<shape.w; w++)
			{
				*p_out = p_in[(h*shape.w + w)*shape.c + c];	
				p_out++;
			}
		}
	}
}

// change format from CHW to HWC
// the shape of the data, input data, output data
void chw2hwc_q7(nnom_shape_t shape, q7_t* p_in, q7_t* p_out)
{
	int im_size = shape.w * shape.h;
	int h_step;
	
	for(int h=0; h<shape.h; h++)
	{
		h_step = shape.w * h;
		for(int w=0; w<shape.w; w++)
		{
			for(int c=0; c<shape.c; c++)
			{
				*p_out = p_in[im_size * c + h_step + w];	
				p_out++;
			}
		}
	}
}

nnom_status_t flatten_run(nnom_layer_t *layer)
{
	#ifdef NNOM_USING_CHW
	// CHW format must reorder to HWC for dense layer and all other 1D layer (?)
	chw2hwc_q7(layer->in->shape, layer->in->mem->blk, layer->out->mem->blk);
	#endif
	// you must be kidding me
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
		if (cl->filter_mult != 1 || layer->in->shape.c % 2 != 0 || layer->out->shape.c % 2)
			return NN_ARGUMENT_ERROR;
		result = (nnom_status_t)arm_depthwise_separable_conv_HWC_q7_nonsquare(
	#else
		local_depthwise_separable_conv_HWC_q7_nonsquare(
	#endif
#endif
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

#ifdef NNOM_USING_CHW
	// CHW format
	local_convolve_CHW_q7_nonsquare(
				layer->in->mem->blk,
				layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
				cl->weights->p_value, layer->out->shape.c,
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_value, cl->bias_shift, cl->output_shift,
				layer->out->mem->blk,
				layer->out->shape.w, layer->out->shape.h, (q15_t *)(layer->comp->mem->blk), NULL);
	return NN_SUCCESS;
#else
	// HWC format
	#ifdef NNOM_USING_CMSIS_NN
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
	// end of cmsis nn
	#else
	// local implementation
	local_convolve_HWC_q7_nonsquare(
				layer->in->mem->blk,
				layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
				cl->weights->p_value, layer->out->shape.c,
				cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h,
				cl->bias->p_value, cl->bias_shift, cl->output_shift,
				layer->out->mem->blk,
				layer->out->shape.w, layer->out->shape.h, (q15_t *)(layer->comp->mem->blk), NULL);
	return NN_SUCCESS;
	#endif
#endif // end of CHW/HWC
}


nnom_status_t zero_padding_run(nnom_layer_t * layer)
{
	nnom_zero_padding_layer_t *cl = (nnom_zero_padding_layer_t*)layer;
	
#ifdef NNOM_USING_CHW
	local_zero_padding_CHW_q7(
#else
	local_zero_padding_HWC_q7(
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

nnom_status_t cell_simple_rnn_run(nnom_layer_t *layer)
{
	/*
	nnom_status_t result;
	// cell / layer
	nnom_rnn_layer_t* cl 	= (nnom_rnn_layer_t *)layer;
	nnom_simple_rnn_cell_t* cell = (nnom_simple_rnn_cell_t*)cl->cell;
	// parameters
	size_t input_size 		= layer->in->shape.c;				// in rnn, h = 1, w = timestamp, c = feature size. 
	size_t output_size 		= cell->super.units;					// output size = state size in keras. 
	q7_t* weight 			= (q7_t*)cell->weights->p_value;
	q7_t* re_weight 		= (q7_t*)cell->weights->p_value + input_size;
	q7_t* bias				= (q7_t*)cell->bias->p_value;
	q7_t* bias_dummy		= (q7_t*)cell->bias->p_value + output_size;// this must be a dummy bias for all zero. 
	uint16_t bias_shift 	= cell->bias->shift; 			 	// not sure
	uint16_t output_shift 	= cell->weights->shift; 			// not correct
	uint8_t* vector_buf 	= layer->comp->mem->blk;			// not correct, buf for calculation
	
	// layer->comp buf is use to store states and intermmediate buffer
	// state buf | B1	|compute buf;  Additionaly, cell->output buffer can be used for calulation
	// 
	// h = tanh or relu(w*x + b_dummy + h*x + bias)

	// w*x + b_dummy
	// buff: input -> B1
	result = (nnom_status_t)arm_fully_connected_q7(
		cell->super.input_buf,
		weight,
		input_size, output_size,
		bias_shift, output_shift,
		bias_dummy,
		cell->super.output_buf, (q15_t*)vector_buf);
	
	// h*x + bias (paramters are wrong)
	// buff: state -> output
	result = (nnom_status_t)arm_fully_connected_q7(
		cell->super.input_buf,
		re_weight,
		input_size, output_size,
		bias_shift, output_shift,
		bias,
		cell->super.output_buf, (q15_t*)vector_buf);
	
	// add (paramters are wrong)
	// buff: B1 + output -> state 
	arm_add_q7(layer->in->mem->blk, layer->out->mem->blk, layer->out->mem->blk, output_size);
	
	// finally the activation (thinking of changing the activation's run interfaces. )
	// buff: state
	result = act_direct_run(layer, cell->activation,  cell->super.output_buf, output_size, layer->in->qfmt);

	// copy to output
	//memcpy(cell->super.output_buf, state, output_size);

	*/
	return NN_SUCCESS;
}

nnom_status_t rnn_run(nnom_layer_t *layer)
{
	nnom_status_t result;
	nnom_rnn_layer_t *cl = (nnom_rnn_layer_t *)(layer);
	size_t timestamps_size = layer->in->shape.w;
	size_t feature_size    = layer->in->shape.c;
	size_t output_size     = cl->cell->units;

	// set the state buffer
	cl->cell->state_buf = layer->comp->mem;

	// currently not support stateful. and not support reserved mem block
	if(!cl->stateful)
		memset(cl->cell->state_buf, 0, shape_size(&layer->comp->shape));

	// run
	for (uint32_t round = 0; round < timestamps_size; round++)
	{
		// set input buffer
		cl->cell->input_buf = (q7_t*)layer->in->mem->blk + feature_size * round;
		if(cl->return_sequence)
			cl->cell->output_buf = (q7_t*)layer->out->mem->blk + output_size * round;
		else
			cl->cell->output_buf = layer->out->mem->blk;

		// run it
		result = cl->cell->run(layer);
	}
	return result;
}

nnom_status_t dense_run(nnom_layer_t *layer)
{
	nnom_status_t result = NN_SUCCESS;
	nnom_dense_layer_t *cl = (nnom_dense_layer_t *)(layer);

#if !(DENSE_WEIGHT_OPT)
	#ifdef NNOM_USING_CMSIS_NN
		result = (nnom_status_t)arm_fully_connected_q7(
	#else
		local_fully_connected_q7(
	#endif
			layer->in->mem->blk,
			cl->weights->p_value,
			layer->in->shape.h, layer->out->shape.h,
			cl->bias_shift, cl->output_shift,
			cl->bias->p_value,
			layer->out->mem->blk, (q15_t *)(layer->comp->mem->blk));
#else
	#ifdef NNOM_USING_CMSIS_NN
		result = (nnom_status_t)arm_fully_connected_q7_opt(
	#else
		local_fully_connected_q7_opt(
	#endif
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
	return act_direct_run(layer, cl->act, layer->in->mem->blk, shape_size(&layer->out->shape), layer->in->qfmt);
}

nnom_status_t maxpool_run(nnom_layer_t *layer)
{
	nnom_maxpool_layer_t *cl = (nnom_maxpool_layer_t *)(layer);

#ifdef NNOM_USING_CHW
	local_maxpool_q7_CHW(layer->in->mem->blk, 				
			layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
			cl->kernel.w, cl->kernel.h, 
			cl->pad.w, cl->pad.h,
			cl->stride.w, cl->stride.h,
			layer->out->shape.w, layer->out->shape.h,
			NULL,
			layer->out->mem->blk);
#else //end of CHW
	// HWC
	#ifdef NNOM_USING_CMSIS_NN
	// 2D, square
	if (layer->in->shape.w == layer->in->shape.h &&
		layer->out->shape.w == layer->out->shape.h)
	{
		arm_maxpool_q7_HWC(
			layer->in->mem->blk,
			layer->in->shape.w, layer->in->shape.c,
			cl->kernel.w, cl->pad.w, cl->stride.w,
			layer->out->shape.w,
			NULL,
			layer->out->mem->blk);
	}
	// none square 2D, or 1D
	else
	#endif
	{
		// CMSIS-NN does not support none-square pooling, we have to use local implementation
		local_maxpool_q7_HWC(layer->in->mem->blk, 				
				layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
				cl->kernel.w, cl->kernel.h, 
				cl->pad.w, cl->pad.h,
				cl->stride.w, cl->stride.h,
				layer->out->shape.w, layer->out->shape.h,
				NULL,
				layer->out->mem->blk);
	}
#endif // CHW/HWC
	return NN_SUCCESS;
}

nnom_status_t avgpool_run(nnom_layer_t *layer)
{
	nnom_avgpool_layer_t *cl = (nnom_avgpool_layer_t *)(layer);
	
#ifdef NNOM_USING_CHW
	local_avepool_q7_CHW(layer->in->mem->blk, 				
			layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
			cl->kernel.w, cl->kernel.h, 
			cl->pad.w, cl->pad.h,
			cl->stride.w, cl->stride.h,
			layer->out->shape.w, layer->out->shape.h,
			NULL,
			layer->out->mem->blk);
#else //end of CHW
	#ifdef NNOM_USING_CMSIS_NN
	// 2D, square
	if (layer->in->shape.w == layer->in->shape.h &&
		layer->out->shape.w == layer->out->shape.h)
	{
		arm_avepool_q7_HWC(
			layer->in->mem->blk,
			layer->in->shape.w, layer->in->shape.c,
			cl->kernel.w, cl->pad.w, cl->stride.w,
			layer->out->shape.w,
			layer->comp->mem->blk,
			layer->out->mem->blk);
	}
	// none square 2D, or 1D
	else
	#endif
	{
		// CMSIS-NN does not support none-square pooling, we have to use local implementation
		local_avepool_q7_HWC(layer->in->mem->blk, 				
				layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
				cl->kernel.w, cl->kernel.h, 
				cl->pad.w, cl->pad.h,
				cl->stride.w, cl->stride.h,
				layer->out->shape.w, layer->out->shape.h,
				NULL,
				layer->out->mem->blk);
	}
#endif
	return NN_SUCCESS;
}

// sum pooling, dynamic change Q format, must be used in the last layer before softmax in current version
nnom_status_t sumpool_run(nnom_layer_t *layer)
{
	nnom_sumpool_layer_t *cl = (nnom_sumpool_layer_t *)(layer);
	
#ifdef NNOM_USING_CHW
	local_sumpool_q7_CHW(				
#else
	local_sumpool_q7_HWC(
#endif
			layer->in->mem->blk, 				
			layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
			cl->kernel.w, cl->kernel.h, 
			cl->pad.w, cl->pad.h,
			cl->stride.w, cl->stride.h,
			layer->out->shape.w, layer->out->shape.h,
			layer->comp->mem->blk,
			layer->out->mem->blk);

	return NN_SUCCESS;
}

// up sampling, or so called unpooling
nnom_status_t upsample_run(nnom_layer_t *layer)
{
	nnom_upsample_layer_t *cl = (nnom_upsample_layer_t *)(layer);
#ifdef NNOM_USING_CHW
	local_up_sampling_q7_CHW(				
#else
	local_up_sampling_q7_HWC(
#endif
			layer->in->mem->blk, 				
			layer->in->shape.w, layer->in->shape.h, layer->in->shape.c,
			cl->kernel.w, cl->kernel.h, 
			layer->out->shape.w, layer->out->shape.h,
			NULL,
			layer->out->mem->blk);

	return NN_SUCCESS;
}


nnom_status_t softmax_run(nnom_layer_t *layer)
{
#ifdef NNOM_USING_CMSIS_NN
	// temporary fixed for mutiple dimension input. 
	arm_softmax_q7(layer->in->mem->blk, shape_size(&layer->out->shape), layer->out->mem->blk);
#else
	local_softmax_q7(layer->in->mem->blk, shape_size(&layer->out->shape), layer->out->mem->blk);
#endif
	return NN_SUCCESS;
}

static inline int chw_i(int hwc)
{
	hwc = hwc+1;			
	if(hwc>2) hwc= 0;
	return hwc;
}

static inline int hwc_i(int chw)
{
	chw = chw -1;			
	if(chw<0) chw=2;
	return chw;
}

nnom_status_t concat_run(nnom_layer_t *layer)
{
	// by default, concat layer has mutiple (>=2) input and 1 output.
	nnom_concat_layer_t *cl = (nnom_concat_layer_t *)layer;
	nnom_shape_axis_t *out_shape = (nnom_shape_axis_t *)(&layer->out->shape); // get the shape.axis[0,1,2...] access to shape type
	nnom_shape_axis_t *in_shape;
	nnom_layer_io_t *in;

#ifdef NNOM_USING_CHW
	// Concatenate for HWC	
	uint8_t *pin;
	uint8_t *pout = layer->out->mem->blk;
	uint32_t block_size;
	uint32_t n_block;
	
	// calcualte number of block to concat. the other shapes before the concat axis
	in_shape = (nnom_shape_axis_t*)(&layer->in->shape);
	n_block = 1;
	for(int i= 0; i< chw_i(cl->axis); i++)
	{
		n_block *= in_shape->axis[hwc_i(i)];
	}
	
	// concat all input layers
	for(int i=0; i<n_block; i++)
	{
		in = layer->in;
		while (in != NULL)
		{
			in_shape = (nnom_shape_axis_t*)(&in->shape);
			// the block size of concat data in this layer
			block_size = 1;
			for(int j= 2; j >= chw_i(cl->axis); j--)
				block_size *= in_shape->axis[hwc_i(j)];
			// concat		
			pin = (uint8_t *)in->mem->blk + i * block_size;
			memcpy(pout, pin, block_size);
			pout += block_size;
			in = in->aux;
		}
	}
	
#else // end of HWC concate
	
	// Concatenate for HWC	
	uint8_t *pin;
	uint8_t *pout = layer->out->mem->blk;
	uint32_t block_size;
	uint32_t n_block;
	
	// calcualte number of block to concat. the other shapes before the concat axis
	in_shape = (nnom_shape_axis_t*)(&layer->in->shape);
	n_block = 1;
	for(int i=0; i< cl->axis; i++)
		n_block *= in_shape->axis[i];
	
	// concat all input layers
	for(int i=0; i<n_block; i++)
	{
		in = layer->in;
		while (in != NULL)
		{
			in_shape = (nnom_shape_axis_t*)(&in->shape);
			// the block size of concat data in this layer
			block_size = 1;
			for(int j=cl->axis; j < 3; j++)
				block_size *= in_shape->axis[j];
			// concat		
			pin = (uint8_t *)in->mem->blk + i * block_size;
			memcpy(pout, pin, block_size);
			pout += block_size;
			in = in->aux;
		}
	}
#endif
	return NN_SUCCESS;
}


nnom_status_t add_run(nnom_layer_t *layer)
{
	nnom_matrix_layer_t* cl = (nnom_matrix_layer_t*)layer;
	nnom_layer_io_t *in;
	size_t size = shape_size(&layer->in->shape);
	int32_t oshift = cl->oshift;

	// adding the first 2 matrix
	#ifdef NNOM_USING_CMSIS_NN
	if(oshift == 0)
		arm_add_q7(layer->in->mem->blk, layer->in->aux->mem->blk, layer->out->mem->blk, size);
	else
	#endif
		local_add_q7(layer->in->mem->blk, layer->in->aux->mem->blk, layer->out->mem->blk, oshift, size);

	
	// if there is 3rd or more, we should use 
	if (layer->in->aux->aux != NULL)
	{
		in = layer->in->aux->aux;
		while (in != NULL)
		{
			// adding the first 2 matrix
			#ifdef NNOM_USING_CMSIS_NN
			if(oshift == 0)
				arm_add_q7(in->mem->blk, layer->out->mem->blk, layer->out->mem->blk, size);
			else
			#endif
				local_add_q7(in->mem->blk, layer->out->mem->blk, layer->out->mem->blk, oshift, size);

			in = in->aux;
		}
	}

	return NN_SUCCESS;
}

nnom_status_t sub_run(nnom_layer_t *layer)
{
	nnom_matrix_layer_t* cl = (nnom_matrix_layer_t*)layer;
	nnom_layer_io_t *in;
	size_t size = shape_size(&layer->in->shape);
	int32_t oshift = cl->oshift;

	// the first 2 matrix
	#ifdef NNOM_USING_CMSIS_NN
	if(oshift == 0)
		arm_sub_q7(layer->in->mem->blk, layer->in->aux->mem->blk, layer->out->mem->blk, size);
	else
	#endif
		local_sub_q7(layer->in->mem->blk, layer->in->aux->mem->blk, layer->out->mem->blk, oshift, size);

	// if there is 3rd or more
	if (layer->in->aux->aux != NULL)
	{
		in = layer->in->aux->aux;
		while (in != NULL)
		{
			// adding the first 2 matrix
			#ifdef NNOM_USING_CMSIS_NN
			if(oshift == 0)
				arm_sub_q7(in->mem->blk, layer->out->mem->blk, layer->out->mem->blk, size);
			else
			#endif
				local_sub_q7(in->mem->blk, layer->out->mem->blk, layer->out->mem->blk, oshift, size);

			in = in->aux;
		}
	}
	return NN_SUCCESS;
}

nnom_status_t mult_run(nnom_layer_t *layer)
{
	nnom_matrix_layer_t* cl = (nnom_matrix_layer_t*)layer;
	nnom_layer_io_t *in;
	size_t size = shape_size(&layer->in->shape);
	int32_t oshift = cl->oshift;

	// the first 2 matrix
	#ifdef NNOM_USING_CMSIS_NN
	if(oshift == 0)
		arm_mult_q7(layer->in->mem->blk, layer->in->aux->mem->blk, layer->out->mem->blk, size);
	else
	#endif
		local_mult_q7(layer->in->mem->blk, layer->in->aux->mem->blk, layer->out->mem->blk, oshift, size);
	
	// if there is 3rd or more
	if (layer->in->aux->aux != NULL)
	{
		in = layer->in->aux->aux;
		while (in != NULL)
		{
			// adding the first 2 matrix
			#ifdef NNOM_USING_CMSIS_NN
			if(oshift == 0)
				arm_sub_q7(in->mem->blk, layer->out->mem->blk, layer->out->mem->blk, size);
			else
			#endif
				local_sub_q7(in->mem->blk, layer->out->mem->blk, layer->out->mem->blk, oshift, size);

			in = in->aux;
		}
	}
	return NN_SUCCESS;
}
