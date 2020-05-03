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



#ifdef __cplusplus
}
#endif

#endif /* __NNOM_RNN_H__ */
