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
nnom_activation_t* act_relu(void);
nnom_activation_t* act_softmax(void);
nnom_activation_t* act_sigmoid(void);
nnom_activation_t* act_tanh(void);

#endif

