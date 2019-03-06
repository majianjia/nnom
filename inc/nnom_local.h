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

#ifndef __NNOM_LOCAL_H__
#define __NNOM_LOCAL_H__

#include "arm_math.h"
#include "arm_nnfunctions.h"

void arm_maxpool_1d_q7_HWC(q7_t *Im_in,
                           const uint16_t dim_im_in,
                           const uint16_t ch_im_in,
                           const uint16_t dim_kernel,
                           const uint16_t padding,
                           const uint16_t stride, const uint16_t dim_im_out, q7_t *bufferA, q7_t *Im_out);

void arm_avepool_1d_q7_HWC(q7_t *Im_in,
                           const uint16_t dim_im_in,
                           const uint16_t ch_im_in,
                           const uint16_t dim_kernel,
                           const uint16_t padding,
                           const uint16_t stride, const uint16_t dim_im_out, q7_t *bufferA, q7_t *Im_out);

// modified from CMSIS-NN test_ref
void local_avepool_q7_HWC(const q7_t * Im_in, // input image
                            const uint16_t dim_im_in_x,   	// input image dimension x or W
							const uint16_t dim_im_in_y,   	// input image dimension y or H
                            const uint16_t ch_im_in,    	// number of input image channels
                            const uint16_t dim_kernel_x,  	// window kernel size
							const uint16_t dim_kernel_y,  	// window kernel size
                            const uint16_t padding_x, 		// padding sizes
							const uint16_t padding_y, 		// padding sizes
                            const uint16_t stride_x,  		// stride
							const uint16_t stride_y,  		// stride
                            const uint16_t dim_im_out_x,  	// output image dimension x or W
							const uint16_t dim_im_out_y,  	// output image dimension y or H
                            q7_t * bufferA, 				// a buffer for local storage, NULL by now
                            q7_t * Im_out);

// modified from CMSIS-NN test_ref                            
void local_maxpool_q7_HWC(const q7_t * Im_in, 				// input image
                            const uint16_t dim_im_in_x,   	// input image dimension x or W
							const uint16_t dim_im_in_y,   	// input image dimension y or H
                            const uint16_t ch_im_in,    	// number of input image channels
                            const uint16_t dim_kernel_x,  	// window kernel size
							const uint16_t dim_kernel_y,  	// window kernel size
                            const uint16_t padding_x, 		// padding sizes
							const uint16_t padding_y, 		// padding sizes
                            const uint16_t stride_x,  		// stride
							const uint16_t stride_y,  		// stride
                            const uint16_t dim_im_out_x,  	// output image dimension x or W
							const uint16_t dim_im_out_y,  	// output image dimension y or H
                            q7_t * bufferA, 				// a buffer for local storage, NULL by now
                            q7_t * Im_out);

#endif
