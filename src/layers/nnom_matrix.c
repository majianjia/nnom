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

// TODO, completely change this file to local version
#ifdef NNOM_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

nnom_layer_t* _same_shape_matrix_layer();
nnom_status_t add_run(nnom_layer_t *layer);
nnom_status_t sub_run(nnom_layer_t *layer);
nnom_status_t mult_run(nnom_layer_t *layer);

nnom_layer_t *Add(int32_t oshift)
{
	nnom_layer_t *layer = _same_shape_matrix_layer();
	nnom_matrix_layer_t *cl = (nnom_matrix_layer_t*) layer;
	if (layer == NULL)
		return NULL;
	// set type in layer parent
	layer->type = NNOM_ADD;
	layer->run = add_run;
	cl->oshift = oshift;
	return layer;
}

nnom_layer_t *Sub(int32_t oshift)
{
	nnom_layer_t *layer = _same_shape_matrix_layer();
	nnom_matrix_layer_t *cl = (nnom_matrix_layer_t*) layer;
	if (layer == NULL)
		return NULL;
	// set type in layer parent
	layer->type = NNOM_SUB;
	layer->run = sub_run;
	cl->oshift = oshift;
	return layer;
}

nnom_layer_t *Mult(int32_t oshift)
{
	nnom_layer_t *layer = _same_shape_matrix_layer();
	nnom_matrix_layer_t *cl = (nnom_matrix_layer_t*) layer;
	if (layer == NULL)
		return NULL;
	// set type in layer parent
	layer->type = NNOM_MULT;
	layer->run = mult_run;
	cl->oshift = oshift;
	return layer;
}


// init a base layer instance with same shape 1 in 1 out. More IO can be added later
// mainly used by matrix calculation (add, mult, sub)
nnom_layer_t *_same_shape_matrix_layer()
{
	nnom_matrix_layer_t *layer;
	nnom_layer_io_t *in, *out;
	//nnom_buf_t *comp;
	size_t mem_size;

	// apply a block memory for all the sub handles.
	mem_size = sizeof(nnom_matrix_layer_t) + sizeof(nnom_layer_io_t) * 2;// + sizeof(nnom_buf_t);
	layer = nnom_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(nnom_matrix_layer_t));
	out = (void *)((uint8_t*)in + sizeof(nnom_layer_io_t));
	//comp = (void *)((uint8_t*)out + sizeof(nnom_layer_io_t));

	// set type in layer parent
	layer->super.build = default_build;
	// set buf state
	in->type = LAYER_BUF_TEMP;
	out->type = LAYER_BUF_TEMP;
	//comp->type = LAYER_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	//layer->super.comp = comp;
	return (nnom_layer_t*)layer;
}



nnom_status_t add_run(nnom_layer_t *layer)
{
	nnom_matrix_layer_t* cl = (nnom_matrix_layer_t*)layer;
	nnom_layer_io_t *in;
	size_t size = tensor_size(layer->in->tensor);
	int32_t oshift = cl->oshift;

	// adding the first 2 matrix
	#ifdef NNOM_USING_CMSIS_NN
	if(oshift == 0)
		arm_add_q7(layer->in->tensor->p_data, layer->in->aux->tensor->p_data, layer->out->tensor->p_data, size);
	else
	#endif
		local_add_q7(layer->in->tensor->p_data, layer->in->aux->tensor->p_data, layer->out->tensor->p_data, oshift, size);

	// if there is 3rd or more, we should use 
	if (layer->in->aux->aux != NULL)
	{
		in = layer->in->aux->aux;
		while (in != NULL)
		{
			// adding the first 2 matrix
			#ifdef NNOM_USING_CMSIS_NN
			if(oshift == 0)
				arm_add_q7(in->tensor->p_data, layer->out->tensor->p_data, layer->out->tensor->p_data, size);
			else
			#endif
				local_add_q7(in->tensor->p_data, layer->out->tensor->p_data, layer->out->tensor->p_data, oshift, size);

			in = in->aux;
		}
	}

	return NN_SUCCESS;
}

nnom_status_t sub_run(nnom_layer_t *layer)
{
	nnom_matrix_layer_t* cl = (nnom_matrix_layer_t*)layer;
	nnom_layer_io_t *in;
	size_t size = tensor_size(layer->in->tensor);
	int32_t oshift = cl->oshift;

	// the first 2 matrix
	#ifdef NNOM_USING_CMSIS_NN
	if(oshift == 0)
		arm_sub_q7(layer->in->tensor->p_data, layer->in->aux->tensor->p_data, layer->out->tensor->p_data, size);
	else
	#endif
		local_sub_q7(layer->in->tensor->p_data, layer->in->aux->tensor->p_data, layer->out->tensor->p_data, oshift, size);

	// if there is 3rd or more
	if (layer->in->aux->aux != NULL)
	{
		in = layer->in->aux->aux;
		while (in != NULL)
		{
			// adding the first 2 matrix
			#ifdef NNOM_USING_CMSIS_NN
			if(oshift == 0)
				arm_sub_q7(in->tensor->p_data, layer->out->tensor->p_data, layer->out->tensor->p_data, size);
			else
			#endif
				local_sub_q7(in->tensor->p_data, layer->out->tensor->p_data, layer->out->tensor->p_data, oshift, size);

			in = in->aux;
		}
	}
	return NN_SUCCESS;
}

nnom_status_t mult_run(nnom_layer_t *layer)
{
	nnom_matrix_layer_t* cl = (nnom_matrix_layer_t*)layer;
	nnom_layer_io_t *in;
	size_t size = size = tensor_size(layer->in->tensor);
	int32_t oshift = cl->oshift;

	// the first 2 matrix
	#ifdef NNOM_USING_CMSIS_NN
	if(oshift == 0)
		arm_mult_q7(layer->in->tensor->p_data, layer->in->aux->tensor->p_data, layer->out->tensor->p_data, size);
	else
	#endif
		local_mult_q7(layer->in->tensor->p_data, layer->in->aux->tensor->p_data, layer->out->tensor->p_data, oshift, size);
	
	// if there is 3rd or more
	if (layer->in->aux->aux != NULL)
	{
		in = layer->in->aux->aux;
		while (in != NULL)
		{
			// adding the first 2 matrix
			#ifdef NNOM_USING_CMSIS_NN
			if(oshift == 0)
				arm_sub_q7(in->tensor->p_data, layer->out->tensor->p_data, layer->out->tensor->p_data, size);
			else
			#endif
				local_sub_q7(in->tensor->p_data, layer->out->tensor->p_data, layer->out->tensor->p_data, oshift, size);

			in = in->aux;
		}
	}
	return NN_SUCCESS;
}
