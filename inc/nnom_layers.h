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


#ifndef __NNOM_LAYERS_H__
#define __NNOM_LAYERS_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_run.h"
#include "nnom_activations.h"

// child layers parameters
typedef struct _nnom_conv2d_layer_t {
	nnom_layer_t super;
	nnom_shape_t kernel;
	nnom_shape_t stride;
	int8_t output_shift;
	int8_t bias_shift;
	nnom_shape_t pad;
	nnom_padding_t padding_type; 
	uint32_t filter_mult;			// filter size (for conv) or multilplier (for depthwise)
	nnom_weight_t *weights;
	nnom_bias_t *bias;
}nnom_conv2d_layer_t;

typedef struct _nnom_dense_layer_t {
	nnom_layer_t super;
	size_t output_unit;
	nnom_weight_t *weights;
	nnom_bias_t *bias;
	int8_t output_shift;
	int8_t bias_shift;

}nnom_dense_layer_t;

// lambda layer
typedef struct _nnom_lambda_layer_t {
	nnom_layer_t super;
	nnom_status_t(*run)(nnom_layer_t *layer);		// 
	nnom_status_t(*oshape)(nnom_layer_t *layer);	// equal to other layer's xxx_output_shape() method, which is to calculate the output shape.
	void* parameters;								// parameters for lambda
}nnom_lambda_layer_t;

// activation layer
typedef struct _nnom_activation_layer_t {
	nnom_layer_t super;
	nnom_activation_t *act;							//
}nnom_activation_layer_t;

// RNN
typedef struct _nnom_rnn_cell_t {
	nnom_status_t(*run)(nnom_layer_t *layer);  	// simple RNN, GRU, LSTM runner
	void* input_buf;							// the input buf and output buf for current cell. 
	void* output_buf;							// These will be set in rnn_run() before entre cell.run()
	size_t unit; 								// 
}nnom_rnn_cell_t;

typedef struct _nnom_simple_rnn_cell_t
{
	nnom_rnn_cell_t super;
	//nnom_status_t(*activation)(nnom_layer_t *layer);
	uint32_t activation;
	
	nnom_weight_t *weights;
	nnom_bias_t *bias;
}nnom_simple_rnn_cell_t;

typedef struct _nnom_gru_cell_t
{
	nnom_rnn_cell_t super;
	uint32_t activation;
	uint32_t recurrent_activation;
	//nnom_status_t(*activation)(nnom_layer_t *layer);
	//nnom_status_t(*activation)(nnom_layer_t *layer);
	
	nnom_weight_t *weights;
	nnom_bias_t *bias;
}nnom_gru_cell_t;

typedef struct _nnom_rnn_layer {
	nnom_layer_t super;
	nnom_rnn_cell_t *cell;
	
	bool return_sequence;				// return sequence?
}nnom_rnn_layer_t;

// Max Pooling
typedef struct _nnom_maxpool_layer {
	nnom_layer_t super;
	nnom_shape_t kernel;
	nnom_shape_t stride;
	nnom_shape_t pad;
	nnom_padding_t padding_type; 
}nnom_maxpool_layer_t;
// Avg pooling
typedef nnom_maxpool_layer_t nnom_avgpool_layer_t;

typedef struct _nnom_io_layer {
	nnom_layer_t super;
	nnom_shape_t shape;
	nnom_qformat_t format;
	void* buf;				//input or output
}nnom_io_layer_t;

typedef struct _nnom_concat_layer {
	nnom_layer_t super;
	int8_t axis;
}nnom_concat_layer_t;

// properties 
nnom_shape_t shape(size_t w, size_t h, size_t c);
nnom_shape_t kernel(size_t w, size_t h);
nnom_shape_t stride(size_t w, size_t h);
nnom_qformat_t qformat(int8_t n, int8_t m);
size_t shape_size(nnom_shape_t *s);

// utils
// this function is to add a new IO to current inited IO 
// input, the targeted IO that the new IO will be added to 
// output , the new IO
nnom_layer_io_t * io_add_aux(nnom_layer_io_t * targeted_io);

// Layer APIs ******

// input/output
nnom_layer_t* Input(nnom_shape_t input_shape, nnom_qformat_t fmt, void* p_buf);
nnom_layer_t* Output(nnom_shape_t output_shape, nnom_qformat_t fmt, void* p_buf);

// Pooling
nnom_layer_t* MaxPool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad);
nnom_layer_t* AvePool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad);

// Activation
nnom_layer_t* Activation(nnom_activation_t *act);
nnom_layer_t* ReLU(void);
nnom_layer_t* Softmax(void);
nnom_layer_t* Sigmoid(void);
nnom_layer_t* TanH(void);


// Matrix
nnom_layer_t * Add(void);
nnom_layer_t * Sub(void);
nnom_layer_t * Mult(void);

// utils
nnom_layer_t* Flatten(void);
nnom_layer_t* Concat(int8_t axis);

// NN Constructers
// conv2d
nnom_layer_t* Conv2D(uint32_t filters, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad,
	nnom_weight_t *w, nnom_bias_t *b);

// depthwise_convolution
nnom_layer_t* DW_Conv2D(uint32_t multiplier, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad, 
	nnom_weight_t *w, nnom_bias_t *b);

// fully connected, dense
nnom_layer_t* Dense(size_t output_unit, nnom_weight_t *w, nnom_bias_t *b);

// rnn layer based
nnom_layer_t* RNN(nnom_rnn_cell_t* cell, bool return_sequence);
// RNN cells
nnom_rnn_cell_t* SimpleCell(size_t unit, uint32_t activation, nnom_weight_t *w, nnom_bias_t *b);

#endif

