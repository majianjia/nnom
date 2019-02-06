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

#include "arm_math.h"
#include "arm_nnfunctions.h"

// porting
static nnom_status_t relu_run(nnom_layer_t* layer, nnom_activation_t* act)
{
	arm_relu_q7(act->data, act->size);
	return NN_SUCCESS;
}

static nnom_status_t tanh_run(nnom_layer_t* layer, nnom_activation_t* act)
{
	arm_nn_activations_direct_q7(act->data, 
		act->size,
		act->fmt.n,
		ARM_TANH);
	
	return NN_SUCCESS;
}

static nnom_status_t sigmoid_run(nnom_layer_t* layer, nnom_activation_t* act)
{
	arm_nn_activations_direct_q7(act->data, 
		act->size,
		act->fmt.n,
		ARM_SIGMOID);
	return NN_SUCCESS;
}

// 
nnom_activation_t* act_relu(void)
{
	nnom_activation_t * act = nnom_mem(sizeof(nnom_activation_t));
	act->run 	= relu_run;
	act->type 	= ACT_RELU;
	return act;
}

nnom_activation_t*  act_tanh(void)
{
	nnom_activation_t * act = nnom_mem(sizeof(nnom_activation_t));
	act->run 	= tanh_run;
	act->type 	= ACT_TANH;
	return act;
}

nnom_activation_t*  act_sigmoid(void)
{
	nnom_activation_t * act = nnom_mem(sizeof(nnom_activation_t));
	act->run 	= sigmoid_run;
	act->type 	= ACT_SIGMOID;
	return act;
}
















