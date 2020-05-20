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

#ifndef __NNOM_ACTIVATION_H__
#define __NNOM_ACTIVATION_H__

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


// activation layer
typedef struct _nnom_activation_layer_t
{
	nnom_layer_t super;
	nnom_activation_t *act; 
} nnom_activation_layer_t;


#ifdef __cplusplus
}
#endif

#endif /* __NNOM_ACTIVATION_H__ */
