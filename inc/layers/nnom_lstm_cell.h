/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-08-24     Jianjia Ma   The first version
 */

#ifndef __NNOM_LSTM_CELL_H__
#define __NNOM_LSTM_CELL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "nnom_rnn.h"
#include "nnom_activation.h"

// a machine interface for configuration
typedef struct _nnom_lstm_cell_config_t
{
	nnom_layer_config_t super;
	nnom_tensor_t *weights;
	nnom_tensor_t* recurrent_weights;
	nnom_tensor_t *bias;
	nnom_qformat_param_t q_dec_iw, q_dec_hw, q_dec_h;
	nnom_activation_type_t act_type;		// type of the activation
	uint16_t units;
} nnom_lstm_cell_config_t;


typedef struct _nnom_lstm_cell_t
{
	nnom_rnn_cell_t super;
	nnom_activation_type_t act_type;

	nnom_tensor_t* weights;
	nnom_tensor_t* recurrent_weights;
	nnom_tensor_t* bias;

	// experimental, 
	// iw: input x weight
	// hw: hidden state x recurrent weight
	// h: hidden state
	nnom_qformat_param_t q_dec_iw, q_dec_hw, q_dec_h;
	nnom_qformat_param_t oshift_iw, oshift_hw, bias_shift;

	uint16_t vsize; // vector size, the input
} nnom_lstm_cell_t;

// LSTM
nnom_rnn_cell_t *lstm_cell_s(const nnom_lstm_cell_config_t* config);

nnom_status_t lstm_cell_free(nnom_rnn_cell_t* cell);
nnom_status_t lstm_cell_build(nnom_rnn_cell_t* cell);
nnom_status_t lstm_cell_run(nnom_rnn_cell_t* cell);


#ifdef __cplusplus
}
#endif

#endif /* __NNOM_LSTM_CELL_H__ */
