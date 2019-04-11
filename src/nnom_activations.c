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

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "nnom.h"
#include "nnom_activations.h"
#include "nnom_local.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

// porting
static nnom_status_t relu_run(nnom_layer_t *layer, nnom_activation_t *act)
{
	#ifdef NNOM_USING_CMSIS_NN
		arm_relu_q7(act->data, act->size);
	#else
		local_relu_q7(act->data, act->size);
	#endif
	return NN_SUCCESS;
}

static nnom_status_t tanh_run(nnom_layer_t *layer, nnom_activation_t *act)
{
	#ifdef NNOM_USING_CMSIS_NN
		arm_nn_activations_direct_q7(act->data, act->size, act->fmt.n, ARM_TANH);
	#else
		local_tanh_q7(act->data, act->size, act->fmt.n);
	#endif

	return NN_SUCCESS;
}

static nnom_status_t sigmoid_run(nnom_layer_t *layer, nnom_activation_t *act)
{
	#ifdef NNOM_USING_CMSIS_NN
		arm_nn_activations_direct_q7(act->data, act->size, act->fmt.n, ARM_SIGMOID);
	#else
		local_sigmoid_q7(act->data, act->size, act->fmt.n);
	#endif
	
	return NN_SUCCESS;
}

//
nnom_activation_t *act_relu(void)
{
	nnom_activation_t *act = nnom_mem(sizeof(nnom_activation_t));
	act->run = relu_run;
	act->type = ACT_RELU;
	return act;
}

nnom_activation_t *act_tanh(int32_t dec_bit)
{
	nnom_activation_t *act = nnom_mem(sizeof(nnom_activation_t));
	act->run = tanh_run;
	act->type = ACT_TANH;
	act->fmt.n = dec_bit;
	return act;
}

nnom_activation_t *act_sigmoid(int32_t dec_bit)
{
	nnom_activation_t *act = nnom_mem(sizeof(nnom_activation_t));
	act->run = sigmoid_run;
	act->type = ACT_SIGMOID;
	act->fmt.n = dec_bit;
	return act;
}

// a direct api, 
// to run an activation directly by passing parameters
nnom_status_t act_direct_run(nnom_layer_t *layer, nnom_activation_t* act, void* data, size_t size, nnom_qformat_t fmt)
{
	act->data = data;
	act->size = size;
	act->fmt = fmt;
	return act->run(layer, act);
}
