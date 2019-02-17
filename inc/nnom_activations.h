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

#ifndef __NNOM_ACTIVATIONS_H__
#define __NNOM_ACTIVATIONS_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"

// Activation
// Softmax is not considered as activation in NNoM, Softmax is in layer API.
nnom_activation_t *act_relu(void);
nnom_activation_t *act_sigmoid(void);
nnom_activation_t *act_tanh(void);

// direct API
nnom_status_t act_direct_run(nnom_layer_t *layer, nnom_activation_t* act, void* data, size_t size, nnom_qformat_t fmt);

#endif
