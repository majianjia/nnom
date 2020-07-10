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

// method
nnom_status_t activation_run(nnom_layer_t* layer);
nnom_status_t activation_free(nnom_layer_t *layer);

// Layer API
nnom_layer_t *Activation(nnom_activation_t *act);
nnom_layer_t *ReLU(void);
nnom_layer_t *LeakyReLU(float alpha);
nnom_layer_t *Sigmoid(int32_t dec_bit);
nnom_layer_t *TanH(int32_t dec_bit);

// activation takes act instance which is created. therefore, it must be free when activation is deleted.
// this is the callback in layer->free
nnom_status_t activation_free(nnom_layer_t *layer);
nnom_status_t activation_run(nnom_layer_t *layer);

// Activation API. 
nnom_activation_t* act_relu(void);
nnom_activation_t* act_leaky_relu(float alpha);
nnom_activation_t* act_tanh(int32_t dec_bit);
nnom_activation_t* act_sigmoid(int32_t dec_bit);

// a direct api on tensor
nnom_status_t act_tensor_run(nnom_activation_t* act, nnom_tensor_t* tensor);

#ifdef __cplusplus
}
#endif

#endif /* __NNOM_ACTIVATION_H__ */
