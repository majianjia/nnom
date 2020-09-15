

/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
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
#include <math.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_layers.h"
#include "layers/nnom_activation.h"

#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

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
	in->type = NNOM_TENSOR_BUF_TEMP;
	out->type = NNOM_TENSOR_BUF_NULL; // when a layer's io is set to NULL, both will point to same mem.
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

nnom_layer_t *LeakyReLU(float alpha)
{
	nnom_layer_t *layer = Activation(act_leaky_relu(alpha));
	if (layer == NULL)
		return NULL;

	// set type in layer parent
	layer->type = NNOM_LEAKY_RELU;
	return layer;
}

nnom_layer_t *AdvReLU(float alpha, float max, float threshold)
{
	nnom_layer_t *layer = Activation(act_adv_relu(alpha, max, threshold));
	if (layer == NULL)
		return NULL;

	// set type in layer parent
	layer->type = NNOM_ADV_RELU;
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

void act_delete(nnom_activation_t* act){
	nnom_free(act);
}

// activation takes act instance which is created. therefore, it must be free when activation is deleted.
// this is the callback in layer->free
nnom_status_t activation_free(nnom_layer_t *layer)
{
	if(layer)
		act_delete(((nnom_activation_layer_t *)layer)->act);
	return NN_SUCCESS;
}

nnom_status_t activation_run(nnom_layer_t *layer)
{
	nnom_activation_layer_t *cl = (nnom_activation_layer_t *)layer;
	return act_tensor_run(cl->act, layer->in->tensor);
}

// porting
static nnom_status_t relu_run(nnom_activation_t* act)
{
#ifdef NNOM_USING_CMSIS_NN
	arm_relu_q7(act->tensor->p_data, tensor_size(act->tensor));
#else
	local_relu_q7(act->tensor->p_data, tensor_size(act->tensor));
#endif
	return NN_SUCCESS;
}

// leaky relu 
static nnom_status_t leaky_relu_run(nnom_activation_t* act)
{
	nnom_activation_leaky_relu_t* a = (nnom_activation_leaky_relu_t*) act;
	local_leaky_relu_q7(act->tensor->p_data, a->alpha, tensor_size(act->tensor));
	return NN_SUCCESS;
}

// advance relu
static nnom_status_t adv_relu_run(nnom_activation_t* act)
{
	nnom_activation_adv_relu_t* a = (nnom_activation_adv_relu_t*) act;

	// we need to convert float to fixpoint in runtime where we can know the tensor's q format
	q7_t max = 127;
	q7_t threshold = a->threshold * (1<<act->tensor->q_dec[0]);

	if(a->max != 0x7fc00000) // QNAN for None
	{
		if(a->max * (1<<act->tensor->q_dec[0]) < 127)
			max = a->max * (1<<act->tensor->q_dec[0]);
	}

	local_adv_relu_q7(act->tensor->p_data, a->negative_slope, max, threshold, tensor_size(act->tensor));
	return NN_SUCCESS;
}


static nnom_status_t tanh_run(nnom_activation_t* act)
{
	nnom_activation_fixed_q_t * a = (nnom_activation_fixed_q_t*)act;
	uint8_t int_bit = 7 - a->dec_bit;

	// arm version cannot handle int_bit > 3
#ifdef NNOM_USING_CMSIS_NN
	if(act->tensor->q_dec[0] <= 3) 
		arm_nn_activations_direct_q7(act->tensor->p_data, tensor_size(act->tensor), int_bit, ARM_TANH);
	else
#endif
		local_tanh_q7(act->tensor->p_data, tensor_size(act->tensor), int_bit);
	return NN_SUCCESS;
}

static nnom_status_t sigmoid_run( nnom_activation_t* act)
{
	nnom_activation_fixed_q_t * a = (nnom_activation_fixed_q_t*)act;
	uint8_t int_bit = 7 - a->dec_bit;

	// arm version cannot handle int_bit > 3
#ifdef NNOM_USING_CMSIS_NN
	if(act->tensor->q_dec[0] <= 3) 
		arm_nn_activations_direct_q7(act->tensor->p_data, tensor_size(act->tensor), int_bit, ARM_SIGMOID);
	else
#endif
		local_sigmoid_q7(act->tensor->p_data, tensor_size(act->tensor), int_bit);
	return NN_SUCCESS;
}

static nnom_status_t hard_tanh_run( nnom_activation_t* act)
{
	nnom_activation_fixed_q_t * a = (nnom_activation_fixed_q_t*)act;
    local_hard_tanh_q7(act->tensor->p_data, tensor_size(act->tensor), a->dec_bit); // hard take dec bit instead
	return NN_SUCCESS;
}

static nnom_status_t hard_sigmoid_run( nnom_activation_t* act)
{
	nnom_activation_fixed_q_t * a = (nnom_activation_fixed_q_t*)act;
    local_hard_sigmoid_q7(act->tensor->p_data, tensor_size(act->tensor), a->dec_bit); // hard take dec bit instead
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

nnom_activation_t* act_leaky_relu(float alpha)
{
	nnom_activation_leaky_relu_t* act = nnom_mem(sizeof(nnom_activation_leaky_relu_t));
	act->super.run = leaky_relu_run;
	act->super.type = ACT_LEAKY_RELU;
	act->alpha = (q7_t)alpha*128;
	return (nnom_activation_t* )act;
}

nnom_activation_t* act_adv_relu(float negative_slope, float max, float threshold)
{
	nnom_activation_adv_relu_t* act = nnom_mem(sizeof(nnom_activation_adv_relu_t));
	act->super.run = adv_relu_run;
	act->super.type = ACT_ADV_RELU;
	act->negative_slope = (q7_t)negative_slope*128;
	act->max = max;
	act->threshold = threshold;
	return (nnom_activation_t* )act;
}

nnom_activation_t* act_tanh(int32_t dec_bit)
{
	nnom_activation_fixed_q_t* act = nnom_mem(sizeof(nnom_activation_fixed_q_t));
	act->super.run = tanh_run;
	act->super.type = ACT_TANH;
	act->dec_bit = dec_bit;
	return (nnom_activation_t*)act;
}

nnom_activation_t* act_sigmoid(int32_t dec_bit)
{
	nnom_activation_fixed_q_t* act = nnom_mem(sizeof(nnom_activation_fixed_q_t));

	act->super.run = sigmoid_run;
	act->super.type = ACT_SIGMOID;
	act->dec_bit = dec_bit;
	return (nnom_activation_t*)act;
}

nnom_activation_t* act_hard_tanh(int32_t dec_bit)
{
	nnom_activation_fixed_q_t* act = nnom_mem(sizeof(nnom_activation_fixed_q_t));

	act->super.run = hard_tanh_run;
	act->super.type = ACT_HARD_TANH;
	act->dec_bit = dec_bit;
	return (nnom_activation_t*)act;
}

nnom_activation_t* act_hard_sigmoid(int32_t dec_bit)
{
	nnom_activation_fixed_q_t* act = nnom_mem(sizeof(nnom_activation_fixed_q_t));

	act->super.run = hard_sigmoid_run;
	act->super.type = ACT_HARD_SIGMOID;
	act->dec_bit = dec_bit;
	return (nnom_activation_t*)act;
}



// return the decimal bit if the activation will change the q format of the layer. 
int32_t act_get_dec_bit(nnom_activation_type_t type, int32_t dec_bit)
{
	switch(type)
	{
		case ACT_RELU:
		case ACT_LEAKY_RELU:
		case ACT_ADV_RELU:
			break;
		case ACT_TANH:
        case ACT_HARD_TANH:
		case ACT_SIGMOID:
        case ACT_HARD_SIGMOID:
			dec_bit = 7;
		default:break;
	}
	return dec_bit;
}

// a direct api to run activate a tensor
nnom_status_t act_tensor_run(nnom_activation_t* act, nnom_tensor_t* tensor)
{
	act->tensor = tensor;
	return act->run(act);
}
