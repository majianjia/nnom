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

#ifndef __NNOM_LAMBDA_H__
#define __NNOM_LAMBDA_H__

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

#include "layers/nnom_input.h"

// lambda layer
typedef struct _nnom_lambda_layer_t
{
	nnom_layer_t super;
	nnom_status_t (*run)(nnom_layer_t *layer);	  //
	nnom_status_t (*build)(nnom_layer_t *layer);  // equal to other layer's xxx_build() method, which is to calculate the output shape.
	void *parameters;							  // parameters for lambda
} nnom_lambda_layer_t;


#ifdef __cplusplus
}
#endif

#endif /* __NNOM_LAMBDA_H__ */
