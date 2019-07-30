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

#ifndef __NNOM_LAYERS_H__
#define __NNOM_LAYERS_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"

// RNN Options
#define SEQUENCE_RETURN		true
#define SEQUENCE_NO			false
#define STATEFUL			true
#define UN_STATEFUL			false

// child layers parameters
typedef struct _nnom_conv2d_layer_t
{
	nnom_layer_t super;
	nnom_shape_t kernel;
	nnom_shape_t stride;
	int8_t output_shift;
	int8_t bias_shift;
	nnom_shape_t pad;
	nnom_padding_t padding_type;
	uint32_t filter_mult; 							// filter size (for conv) or multilplier (for depthwise)
	const nnom_weight_t *weights;
	const nnom_bias_t *bias;
} nnom_conv2d_layer_t;

typedef struct _nnom_dense_layer_t
{
	nnom_layer_t super;
	size_t output_unit;
	const nnom_weight_t *weights;
	const nnom_bias_t *bias;
	int8_t output_shift;
	int8_t bias_shift;

} nnom_dense_layer_t;

// zero padding
typedef struct _nnom_zero_padding_layer_t
{
	nnom_layer_t super;
	nnom_border_t pad;
} nnom_zero_padding_layer_t;

// Cropping, same as zeropadding
typedef nnom_zero_padding_layer_t nnom_cropping_layer_t;

// lambda layer
typedef struct _nnom_lambda_layer_t
{
	nnom_layer_t super;
	nnom_status_t (*run)(nnom_layer_t *layer);	  //
	nnom_status_t (*build)(nnom_layer_t *layer);  // equal to other layer's xxx_build() method, which is to calculate the output shape.
	void *parameters;							  // parameters for lambda
} nnom_lambda_layer_t;

// activation layer
typedef struct _nnom_activation_layer_t
{
	nnom_layer_t super;
	nnom_activation_t *act; 
} nnom_activation_layer_t;

// matrix layer
typedef struct _nnom_matrix_layer_t
{
	nnom_layer_t super;
	int32_t oshift;		// output right shift
} nnom_matrix_layer_t;

// RNN
typedef struct _nnom_rnn_cell_t
{
	nnom_status_t (*run)(nnom_layer_t *layer); // simple RNN, GRU, LSTM runner
	void *input_buf;						   // the input buf and output buf for current cell.
	void *output_buf;						   // These will be set in rnn_run() before entre cell.run()
	void *state_buf;						   // state
	size_t units;							   //
} nnom_rnn_cell_t;

typedef struct _nnom_simple_rnn_cell_t
{
	nnom_rnn_cell_t super;
	nnom_activation_t* activation;

	const nnom_weight_t *weights;
	const nnom_bias_t *bias;
} nnom_simple_rnn_cell_t;

typedef struct _nnom_gru_cell_t
{
	nnom_rnn_cell_t super;
	nnom_activation_t* activation;
	nnom_activation_t* recurrent_activation;
	//nnom_status_t(*activation)(nnom_layer_t *layer);
	//nnom_status_t(*activation)(nnom_layer_t *layer);

	const nnom_weight_t *weights;
	const nnom_bias_t *bias;
} nnom_gru_cell_t;

typedef struct _nnom_rnn_layer_t
{
	nnom_layer_t super;
	nnom_rnn_cell_t *cell;

	bool return_sequence; // return sequence?
	bool stateful;
} nnom_rnn_layer_t;

// Max Pooling
typedef struct _nnom_maxpool_layer_t
{
	nnom_layer_t super;
	nnom_shape_t kernel;
	nnom_shape_t stride;
	nnom_shape_t pad;
	nnom_padding_t padding_type;
	int16_t output_shift;			// reserve
} nnom_maxpool_layer_t;

// Avg Pooling
typedef nnom_maxpool_layer_t nnom_avgpool_layer_t;

// Sum Pooling
typedef nnom_maxpool_layer_t nnom_sumpool_layer_t;

// Up Sampling layer (UnPooling)
typedef struct _nnom_upsample_layer_t
{
	nnom_layer_t super;
	nnom_shape_t kernel;
} nnom_upsample_layer_t;

// IO layer
typedef struct _nnom_io_layer
{
	nnom_layer_t super;
	nnom_shape_t shape;
	void *buf; //input or output
} nnom_io_layer_t;

// concatenate layer
typedef struct _nnom_concat_layer
{
	nnom_layer_t super;
	int8_t axis;
} nnom_concat_layer_t;

// properties
nnom_shape_t shape(size_t h, size_t w, size_t c);
nnom_shape_t kernel(size_t h, size_t w);
nnom_shape_t stride(size_t h, size_t w);
nnom_border_t border(size_t top, size_t bottom, size_t left, size_t right);
nnom_qformat_t qformat(int8_t m, int8_t n);
size_t shape_size(nnom_shape_t* s);

// this function is to add a new IO to current inited IO
// input, the targeted IO that the new IO will be added to
// output , the new IO
nnom_layer_io_t* io_add_aux(nnom_layer_io_t* targeted_io);
nnom_layer_io_t *io_init(void *owner_layer, nnom_layer_io_t *io);

#define NN_CEILIF(x,y) ((x+y-1)/y)

// Layer APIs ******

// input/output
nnom_layer_t *Input(nnom_shape_t input_shape, void *p_buf);
nnom_layer_t *Output(nnom_shape_t output_shape, void *p_buf);

// Pooling
nnom_layer_t *MaxPool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad);
nnom_layer_t *AvgPool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad);
nnom_layer_t *SumPool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad);
nnom_layer_t *GlobalMaxPool(void);
nnom_layer_t *GlobalAvgPool(void);
nnom_layer_t *GlobalSumPool(void);

// padding, cropping, upsample
nnom_layer_t *UpSample(nnom_shape_t kernel);	
nnom_layer_t *ZeroPadding(nnom_border_t pad);
nnom_layer_t *Cropping(nnom_border_t pad);

// Activation
nnom_layer_t *Activation(nnom_activation_t *act);
nnom_layer_t *ReLU(void);
nnom_layer_t *Softmax(void);
nnom_layer_t *Sigmoid(int32_t dec_bit);  // input dec bit
nnom_layer_t *TanH(int32_t dec_bit);     // input dec bit 

// Matrix
nnom_layer_t *Add(int32_t oshift);       // output shift
nnom_layer_t *Sub(int32_t oshift);       // output shift			
nnom_layer_t *Mult(int32_t oshift);      // output shift

nnom_layer_t *Flatten(void);
nnom_layer_t *Concat(int8_t axis);

// -- NN Constructers --
// conv2d
nnom_layer_t *Conv2D(uint32_t filters, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad,
					 const nnom_weight_t *w, const nnom_bias_t *b);

// depthwise_convolution
nnom_layer_t *DW_Conv2D(uint32_t multiplier, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad,
						const nnom_weight_t *w, const nnom_bias_t *b);

// fully connected, dense
nnom_layer_t *Dense(size_t output_unit, const nnom_weight_t *w, const nnom_bias_t *b);

// rnn layer based
nnom_layer_t *RNN(nnom_rnn_cell_t *cell, bool return_sequence);

// RNN cells
// The shape for RNN input is (batch, timestamp, feature), where batch is always 1. 
//
// SimpleRNNCell
nnom_rnn_cell_t *SimpleCell(size_t units, nnom_activation_t* activation, const nnom_weight_t *w, const nnom_bias_t *b);

// Lambda Layers
nnom_layer_t *Lambda(nnom_status_t (*run)(nnom_layer_t *),	// run method, required
					 nnom_status_t (*build)(nnom_layer_t *), // optional, call default_build() if left null
					 nnom_status_t (*free)(nnom_layer_t *),   // not required if no resources needs to be deleted, can be left null.
					 void *parameters);						  // user private parameters for run method, left null if not needed.


// default
nnom_status_t default_build(nnom_layer_t* layer);
nnom_status_t input_build(nnom_layer_t* layer);

nnom_status_t conv2d_build(nnom_layer_t* layer);
nnom_status_t dw_conv2d_build(nnom_layer_t* layer);
nnom_status_t dense_build(nnom_layer_t* layer);
nnom_status_t rnn_build(nnom_layer_t* layer);

nnom_status_t upsample_build(nnom_layer_t* layer);
nnom_status_t zero_padding_build(nnom_layer_t* layer);
nnom_status_t cropping_build(nnom_layer_t* layer);

nnom_status_t maxpooling_build(nnom_layer_t* layer);
nnom_status_t avgpooling_build(nnom_layer_t* layer);
nnom_status_t sumpooling_build(nnom_layer_t* layer);
nnom_status_t global_pooling_build(nnom_layer_t* layer);

nnom_status_t flatten_build(nnom_layer_t* layer);
nnom_status_t concat_build(nnom_layer_t* layer);

// run
nnom_status_t input_run(nnom_layer_t* layer);
nnom_status_t output_run(nnom_layer_t* layer);
nnom_status_t flatten_run(nnom_layer_t* layer);
nnom_status_t default_run(nnom_layer_t* layer);  // simply copy data from input to output

nnom_status_t dw_conv2d_run(nnom_layer_t* layer);
nnom_status_t conv2d_run(nnom_layer_t* layer);
nnom_status_t dense_run(nnom_layer_t* layer);
nnom_status_t rnn_run(nnom_layer_t* layer);
nnom_status_t cell_simple_rnn_run(nnom_layer_t* layer);

nnom_status_t upsample_run(nnom_layer_t* layer);
nnom_status_t zero_padding_run(nnom_layer_t* layer);
nnom_status_t cropping_run(nnom_layer_t* layer);

nnom_status_t activation_run(nnom_layer_t* layer);
nnom_status_t softmax_run(nnom_layer_t* layer);

nnom_status_t maxpool_run(nnom_layer_t* layer);
nnom_status_t avgpool_run(nnom_layer_t* layer);
nnom_status_t sumpool_run(nnom_layer_t* layer);

nnom_status_t concat_run(nnom_layer_t* layer);
nnom_status_t add_run(nnom_layer_t* layer);
nnom_status_t sub_run(nnom_layer_t* layer);
nnom_status_t mult_run(nnom_layer_t* layer);


// Activation APIs
// Softmax is not considered as activation in NNoM, Softmax is in layer API.
nnom_activation_t* act_relu(void);
nnom_activation_t* act_sigmoid(int32_t dec_bit);
nnom_activation_t* act_tanh(int32_t dec_bit);

// direct API
nnom_status_t act_direct_run(nnom_activation_t* act, void* data, size_t size, nnom_qformat_t qfmt);
nnom_status_t act_tensor_run(nnom_activation_t* act, nnom_tensor_t* tensor);

#endif
