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
 * 2019-02-10     Jianjia Ma   Compiler supports dense net connection
 */

#ifndef __NNOM_TENSOR_H__
#define __NNOM_TENSOR_H__

#include "nnom.h"

nnom_tensor_t* new_tensor(nnom_tensor_t* t, uint32_t num_dim);
nnom_tensor_t* tensor_set_attribuites(nnom_tensor_t* t, nnom_qformat_t qfmt, uint32_t num_dim, nnom_shape_data_t* dim);
nnom_tensor_t* tensor_cpy_attributes(nnom_tensor_t* des, nnom_tensor_t* src);
size_t tensor_size(nnom_tensor_t* t);


// only support 3d tensor
// change format from CHW to HWC
// the shape of the data, input data, output data
void tensor_hwc2chw_q7(nnom_tensor_t* des, nnom_tensor_t* src);

// change format from CHW to HWC
// the shape of the data, input data, output data
void tensor_chw2hwc_q7(nnom_tensor_t* des, nnom_tensor_t* src);

// deprecated. 
void hwc2chw_q7(nnom_shape_t shape, q7_t* p_in, q7_t* p_out);
void chw2hwc_q7(nnom_shape_t shape, q7_t* p_in, q7_t* p_out);

#endif
