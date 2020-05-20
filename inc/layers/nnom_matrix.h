/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-05-03     Jianjia Ma   The first version
 */

#ifndef __NNOM_MATRIX_H__
#define __NNOM_MATRIX_H__

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

// matrix layer
typedef struct _nnom_matrix_layer_t
{
	nnom_layer_t super;
	int32_t oshift;		// output right shift
} nnom_matrix_layer_t;


#ifdef __cplusplus
}
#endif

#endif /* __NNOM_MATRIX_H__ */
