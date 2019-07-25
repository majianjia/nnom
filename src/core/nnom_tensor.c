/*
 * Copyright (c) 2018-2019
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-02-05     Jianjia Ma   The first version
 * 2019-02-14	  Jianjia Ma   Add layer.free() method.
 */

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>
#include "nnom.h"
#include "nnom_tensor.h"

 // tensor size
size_t tensor_size(nnom_tensor_t* t)
{
	size_t size = 0;
	if (t)
	{
		size = t->dim[0];
		for (int i = 1; i < t->num_dim; i++)
			size *= t->dim[i];
	}
	return size;
}

// initial tensor
nnom_tensor_t* new_tensor(nnom_tensor_t* t, uint32_t num_dim)
{
	if (t)
		nnom_free(t);
	t = nnom_mem(nnom_alignto(sizeof(nnom_tensor_t), 4) + num_dim*sizeof(nnom_shape_data_t));
	t->dim = (nnom_shape_data_t*)((uint8_t*)t + sizeof(nnom_tensor_t));
	return t;
}

// initial tensor
nnom_tensor_t* tensor_set_attribuites(nnom_tensor_t* t, nnom_qformat_t qfmt, uint32_t num_dim, nnom_shape_data_t* dim)
{
	t->qfmt = qfmt;
	t->num_dim = num_dim;
	for (int i = 0; i < num_dim; i++)
		t->dim[i] = dim[i];
	return t;
}


// this method copy the attributes of a tensor to a new tensor
// Note, the tensors must have the same lenght. this method wont cpy the memory pointer data (we will assign memory later after building)
nnom_tensor_t* tensor_cpy_attributes(nnom_tensor_t* des, nnom_tensor_t* src)
{
	des->num_dim = src->num_dim;
	des->qfmt = src->qfmt;
	memcpy(des->dim, src->dim, src->num_dim * sizeof(nnom_shape_data_t));
	return des;
}

// change format from CHW to HWC
// the shape of the data, input data, output data
void tensor_hwc2chw_q7(nnom_tensor_t* des, nnom_tensor_t* src)
{
	q7_t* p_out = des->p_data;
	q7_t* p_in = src->p_data;

	for (int c = 0; c < src->dim[2]; c++)
	{
		for (int h = 0; h < src->dim[0]; h++)
		{
			for (int w = 0; w < src->dim[1]; w++)
			{
				*p_out = p_in[(h * src->dim[1] + w) * src->dim[2] + c];
				p_out++;
			}
		}
	}
}


// only support 3d tensor
// change format from CHW to HWC
// the shape of the data, input data, output data
void tensor_chw2hwc_q7(nnom_tensor_t* des, nnom_tensor_t* src)
{
	q7_t* p_out = des->p_data;
	q7_t* p_in = src->p_data;

	int im_size = 1;
	int h_step;
	h_step = src->dim[0];

	for (int i = 1; i < src->num_dim; i++)
		im_size *= src->dim[i];

	for (int h = 0; h < src->dim[0]; h++)
	{
		h_step = src->dim[1] * h;
		for (int w = 0; w < src->dim[1]; w++)
		{
			// for 3d tensor
			for (int c = 0; c < src->dim[1]; c++)
			{
				*p_out = p_in[im_size * c + h_step + w];
				p_out++;
			}
		}
	}

}

// (deprecated by tensor_hwc2chw version)
// change format from CHW to HWC
// the shape of the data, input data, output data
void hwc2chw_q7(nnom_shape_t shape, q7_t* p_in, q7_t* p_out)
{
	for (int c = 0; c < shape.c; c++)
	{
		for (int h = 0; h < shape.h; h++)
		{
			for (int w = 0; w < shape.w; w++)
			{
				*p_out = p_in[(h * shape.w + w) * shape.c + c];
				p_out++;
			}
		}
	}
}

// (deprecated)
// change format from CHW to HWC
// the shape of the data, input data, output data
void chw2hwc_q7(nnom_shape_t shape, q7_t* p_in, q7_t* p_out)
{
	int im_size = shape.w * shape.h;
	int h_step;

	for (int h = 0; h < shape.h; h++)
	{
		h_step = shape.w * h;
		for (int w = 0; w < shape.w; w++)
		{
			for (int c = 0; c < shape.c; c++)
			{
				*p_out = p_in[im_size * c + h_step + w];
				p_out++;
			}
		}
	}
}