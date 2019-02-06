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


#ifndef __NNOM_OUT_SHAPE_H__
#define __NNOM_OUT_SHAPE_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "nnom.h"

// default 
// restrict to primary i/o only, set the output same as input.  
nnom_status_t default_out_shape(nnom_layer_t * layer);

// IO
nnom_status_t input_out_shape(nnom_layer_t * layer);
nnom_status_t output_out_shape(nnom_layer_t * layer);

// nn
nnom_status_t conv2d_out_shape(nnom_layer_t * layer);
nnom_status_t dw_conv2d_out_shape(nnom_layer_t * layer);
nnom_status_t dense_out_shape(nnom_layer_t * layer);
nnom_status_t rnn_out_shape(nnom_layer_t* layer);

// activation
nnom_status_t relu_out_shape(nnom_layer_t * layer);
nnom_status_t softmax_out_shape(nnom_layer_t * layer);

// pooling
nnom_status_t maxpooling_out_shape(nnom_layer_t * layer);

// utils
nnom_status_t flatten_out_shape(nnom_layer_t * layer);
nnom_status_t concatenate_out_shape(nnom_layer_t * layer);
nnom_status_t same_shape_2in_1out_out_shape(nnom_layer_t * layer);

#endif
