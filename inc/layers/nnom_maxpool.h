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

#ifndef __NNOM_MAXPOOL_H__
#define __NNOM_MAXPOOL_H__

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


// Max Pooling
typedef struct _nnom_maxpool_layer_t
{
	nnom_layer_t super;
	nnom_shape_t kernel;
	nnom_shape_t stride;
	nnom_shape_t pad;
	nnom_padding_t padding_type;
	int16_t output_shift;			// reserve
} nnom_maxpool_layer_t;





#ifdef __cplusplus
}
#endif

#endif /* __NNOM_MATRIX_H__ */
