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


#ifndef __NNOM_RUN_H__
#define __NNOM_RUN_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"

nnom_status_t input_run(nnom_layer_t *layer);
nnom_status_t output_run(nnom_layer_t *layer);
nnom_status_t flatten_run(nnom_layer_t *layer);

nnom_status_t dw_conv2d_run(nnom_layer_t *layer);
nnom_status_t conv2d_run(nnom_layer_t *layer);
nnom_status_t dense_run(nnom_layer_t *layer);
nnom_status_t rnn_run(nnom_layer_t *layer);
nnom_status_t cell_simple_rnn_run(nnom_layer_t *layer);

nnom_status_t activation_run(nnom_layer_t *layer);
nnom_status_t softmax_run(nnom_layer_t *layer);

nnom_status_t maxpool_run(nnom_layer_t *layer);
nnom_status_t avepool_run(nnom_layer_t *layer);
nnom_status_t concat_run(nnom_layer_t *layer);

nnom_status_t add_run(nnom_layer_t *layer);
nnom_status_t sub_run(nnom_layer_t *layer);
nnom_status_t mult_run(nnom_layer_t *layer);

#endif





