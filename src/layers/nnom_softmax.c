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

nnom_status_t softmax_run(nnom_layer_t *layer);

nnom_layer_t *Softmax(void)
{
	nnom_layer_t *layer;
	nnom_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(nnom_layer_t) + sizeof(nnom_layer_io_t) * 2;
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->type = NNOM_SOFTMAX;
	layer->run = softmax_run;
	layer->build = default_build;
	// set buf state
	in->type = LAYER_BUF_TEMP;
	out->type = LAYER_BUF_TEMP;
	// put in & out on the layer.
	layer->in = io_init(layer, in);
	layer->out = io_init(layer, out);

	return layer;
}

nnom_status_t softmax_run(nnom_layer_t *layer)
{
	#ifdef NNOM_USING_CMSIS_NN
	// temporary fixed for mutiple dimension input. 
	arm_softmax_q7(layer->in->tensor->p_data, tensor_size(layer->out->tensor), layer->out->tensor->p_data);
	#else
	local_softmax_q7(layer->in->tensor->p_data, tensor_size(layer->out->tensor), layer->out->tensor->p_data);
	#endif
	return NN_SUCCESS;
}
