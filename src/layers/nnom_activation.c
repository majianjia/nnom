

/*
 * Copyright (c) 2018-2019
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-07-23     Jianjia Ma   The first version
 */

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_layers.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

nnom_status_t activation_run(nnom_layer_t* layer);

// activation takes act instance which is created. therefore, it must be free when activation is deleted.
// this is the callback in layer->free
static nnom_status_t activation_free(nnom_layer_t *layer)
{
	nnom_free(((nnom_activation_layer_t *)layer)->act);
	return NN_SUCCESS;
}

nnom_layer_t *Activation(nnom_activation_t *act)
{
	nnom_activation_layer_t *layer;
	nnom_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_activation_layer_t) + sizeof(nnom_layer_io_t) * 2;
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_activation_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.type = NNOM_ACTIVATION;
	layer->super.run = activation_run;
	layer->super.build = default_build;
	// set buf state
	in->type = LAYER_BUF_TEMP;
	out->type = LAYER_BUF_NULL; // when a layer's io is set to NULL, both will point to same mem.
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);

	// set activation to layer
	layer->act = act;

	// set free method
	layer->super.free = activation_free;

	return (nnom_layer_t *)layer;
}

nnom_layer_t *ReLU(void)
{
	nnom_layer_t *layer = Activation(act_relu());
	if (layer == NULL)
		return NULL;

	// set type in layer parent
	layer->type = NNOM_RELU;
	return layer;
}

nnom_layer_t *Sigmoid(int32_t dec_bit)
{
	nnom_layer_t *layer = Activation(act_sigmoid(dec_bit));
	if (layer == NULL)
		return NULL;

	// set type in layer parent
	layer->type = NNOM_SIGMOID;
	return layer;
}

nnom_layer_t *TanH(int32_t dec_bit)
{
	nnom_layer_t *layer = Activation(act_tanh(dec_bit));
	if (layer == NULL)
		return NULL;
	// set type in layer parent
	layer->type = NNOM_TANH;
	return layer;
}

nnom_status_t activation_run(nnom_layer_t *layer)
{
	nnom_activation_layer_t *cl = (nnom_activation_layer_t *)layer;
	return act_tensor_run(cl->act, layer->in->tensor);
}


#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

// porting
static nnom_status_t relu_run(nnom_activation_t* act)
{
#ifdef NNOM_USING_CMSIS_NN
	arm_relu_q7(act->data, act->size);
#else
	local_relu_q7(act->data, act->size);
#endif
	return NN_SUCCESS;
}

static nnom_status_t tanh_run(nnom_activation_t* act)
{
	// arm version cannot handle int_bit > 3
#ifdef NNOM_USING_CMSIS_NN
	if (act->qfmt.m <= 3)
		arm_nn_activations_direct_q7(act->data, act->size, act->qfmt.m, ARM_TANH);
	else
#endif
		local_tanh_q7(act->data, act->size, act->qfmt.m);

	return NN_SUCCESS;
}

static nnom_status_t sigmoid_run( nnom_activation_t* act)
{
	// arm version cannot handle int_bit > 3
#ifdef NNOM_USING_CMSIS_NN
	if (act->qfmt.m <= 3)
		arm_nn_activations_direct_q7(act->data, act->size, act->qfmt.m, ARM_SIGMOID);
	else
#endif
		local_sigmoid_q7(act->data, act->size, act->qfmt.m);
	return NN_SUCCESS;
}

//
nnom_activation_t* act_relu(void)
{
	nnom_activation_t* act = nnom_mem(sizeof(nnom_activation_t));
	act->run = relu_run;
	act->type = ACT_RELU;
	return act;
}

nnom_activation_t* act_tanh(int32_t dec_bit)
{
	nnom_activation_t* act = nnom_mem(sizeof(nnom_activation_t));
	act->run = tanh_run;
	act->type = ACT_TANH;
	act->qfmt.n = dec_bit;
	act->qfmt.m = 7 - dec_bit;
	return act;
}

nnom_activation_t* act_sigmoid(int32_t dec_bit)
{
	nnom_activation_t* act = nnom_mem(sizeof(nnom_activation_t));
	act->run = sigmoid_run;
	act->type = ACT_SIGMOID;
	act->qfmt.n = dec_bit;
	act->qfmt.m = 7 - dec_bit;
	return act;
}

// a direct api, 
// to run an activation directly by passing parameters
nnom_status_t act_direct_run(nnom_activation_t* act, void* data, size_t size, nnom_qformat_t qfmt)
{
	act->data = data;
	act->size = size;
	act->qfmt = qfmt;
	return act->run(act);
}

// a direct api on tensor
// a activate a tensor
nnom_status_t act_tensor_run(nnom_activation_t* act, nnom_tensor_t* tensor)
{
	act->data = tensor->p_data;
	act->size = tensor_size(tensor);
	act->qfmt = tensor->qfmt;
	return act->run(act);
}
