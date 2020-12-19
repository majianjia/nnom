/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-12-07     Jianjia Ma   The first version
 */

#ifndef __NNOM_RESHAPE_H__
#define __NNOM_RESHAPE_H__

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

typedef struct _nnom_reshape_layer_t
{
	nnom_layer_t super;
	nnom_shape_data_t* dim;
    uint8_t num_dim;

} nnom_reshape_layer_t;

typedef struct nnom_reshape_config_t
{
	nnom_layer_config_t super;
	nnom_shape_data_t* dim;
    uint8_t num_dim;
} nnom_reshape_config_t;

// method
nnom_status_t reshape_run(nnom_layer_t *layer);
nnom_status_t reshape_build(nnom_layer_t *layer);
nnom_status_t reshape_free(nnom_layer_t *layer);

// API
nnom_layer_t *reshape_s(const nnom_reshape_config_t *config);

#ifdef __cplusplus
}
#endif

#endif /* __NNOM_CONV2D_H__ */
