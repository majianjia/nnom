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

#ifndef __NNOM_ACTIVATIONS_H__
#define __NNOM_ACTIVATIONS_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"

// Activation
// Softmax is not considered as activation in NNoM, Softmax is in layer API.
nnom_activation_t *act_relu(void);
nnom_activation_t *act_sigmoid(int32_t dec_bit);
nnom_activation_t *act_tanh(int32_t dec_bit);

// direct API
nnom_status_t act_direct_run(nnom_layer_t *layer, nnom_activation_t *act, void *data, size_t size, nnom_qformat_t fmt);

#endif
