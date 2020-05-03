/*
 * Copyright (c) 2018-2020
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-05-03     Jianjia Ma   The first version
 */

#ifndef __NNOM_RNN_H__
#define __NNOM_RNN_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_layers.h"
#include "nnom_local.h"
#include "nnom_tensor.h"

// RNN Options
#define SEQUENCE_RETURN		true
#define SEQUENCE_NO			false
#define STATEFUL			true
#define UN_STATEFUL			false

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


// rnn layer based
nnom_layer_t *RNN(nnom_rnn_cell_t *cell, bool return_sequence);

// RNN cells
// The shape for RNN input is (batch, timestamp, feature), where batch is always 1. 
//
// SimpleRNNCell
nnom_rnn_cell_t *SimpleCell(size_t units, nnom_activation_t* activation, const nnom_weight_t *w, const nnom_bias_t *b);

#ifdef __cplusplus
}
#endif

#endif /* __NNOM_RNN_H__ */
