/*
 * Copyright (c) 2018-2019
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Notice: 
 * Code in this file inlcudes derivative works from CMSIS, which is released under alternative license.
 * Please check the LICENSE file for detial.
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-02-05     Jianjia Ma   The first version
 * 2019-03-19     Jianjia Ma   Local C implementation partly from CMSIS-NN
 */

#ifndef __NNOM_LOCAL_H__
#define __NNOM_LOCAL_H__

#include "stdint.h"
#include "nnom.h"
#include "nnom_port.h"

// no idea what is it 
#ifdef ARM_NN_TRUNCATE
#define NNOM_TRUNCATE
#endif

// SSAT implementation with C code
#ifndef __NNOM_SSAT
static inline int __NNOM_SSAT(int32_t value, int32_t bit) {
    int32_t min = -(1<<(bit-1));
    int32_t max = (1<<(bit-1)) - 1;
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return value;
}
#endif

// USAT implementation with C code
#ifndef __NNOM_USAT
static inline int __NNOM_USAT(int32_t value, int32_t bit) {
    int32_t max = (1<<(bit-1)) - 1;
    if (value < 0)
        return 0;
    else if (value > max)
        return max;
    else
        return value;
}
#endif


// Those functions/tables below are partially modifed from CMSIS-NN lib
// https://github.com/ARM-software/CMSIS_5
//
void local_avepool_q7_HWC(const q7_t *Im_in,           // input image
	const uint16_t dim_im_in_x,  // input image dimension x or W
	const uint16_t dim_im_in_y,  // input image dimension y or H
	const uint16_t ch_im_in,     // number of input image channels
	const uint16_t dim_kernel_x, // window kernel size
	const uint16_t dim_kernel_y, // window kernel size
	const uint16_t padding_x,    // padding sizes
	const uint16_t padding_y,    // padding sizes
	const uint16_t stride_x,     // stride
	const uint16_t stride_y,     // stride
	const uint16_t dim_im_out_x, // output image dimension x or W
	const uint16_t dim_im_out_y, // output image dimension y or H
	const uint16_t output_shift, // output right shift
	q7_t *bufferA,               // a buffer for local storage, NULL by now
	q7_t *Im_out);

void local_avepool_q7_CHW(const q7_t *Im_in,           // input image
	const uint16_t dim_im_in_x,  // input image dimension x or W
	const uint16_t dim_im_in_y,  // input image dimension y or H
	const uint16_t ch_im_in,     // number of input image channels
	const uint16_t dim_kernel_x, // window kernel size
	const uint16_t dim_kernel_y, // window kernel size
	const uint16_t padding_x,    // padding sizes
	const uint16_t padding_y,    // padding sizes
	const uint16_t stride_x,     // stride
	const uint16_t stride_y,     // stride
	const uint16_t dim_im_out_x, // output image dimension x or W
	const uint16_t dim_im_out_y, // output image dimension y or H
	const uint16_t output_shift, // output right shift
	q7_t *bufferA,               // a buffer for local storage, NULL by now
	q7_t *Im_out);

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

void local_maxpool_q7_CHW(const q7_t * Im_in, 				// input image
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
							
int32_t local_sumpool_q7_HWC(const q7_t * Im_in, // input image
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
	q7_t * bufferA, 				// a buffer for local storage, size = 4*output_size
	q7_t * Im_out);
							
int32_t local_sumpool_q7_CHW(const q7_t * Im_in, // input image
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
	q7_t * bufferA, 				// a buffer for local storage, size = 4*output_size
	q7_t * Im_out);

// customised up sample pooling
void local_up_sampling_q7_HWC(const q7_t *Im_in,       // input image
	const uint16_t dim_im_in_x,  // input image dimension x or W
	const uint16_t dim_im_in_y,  // input image dimension y or H
	const uint16_t ch_im_in,     // number of input image channels
	const uint16_t dim_kernel_x, // window kernel size
	const uint16_t dim_kernel_y, // window kernel size
	const uint16_t dim_im_out_x, // output image dimension x or W
	const uint16_t dim_im_out_y, // output image dimension y or H
	q7_t *bufferA,               // NULL
	q7_t *Im_out);
						  
void local_up_sampling_q7_CHW(const q7_t *Im_in,       // input image
	const uint16_t dim_im_in_x,  // input image dimension x or W
	const uint16_t dim_im_in_y,  // input image dimension y or H
	const uint16_t ch_im_in,     // number of input image channels
	const uint16_t dim_kernel_x, // window kernel size
	const uint16_t dim_kernel_y, // window kernel size
	const uint16_t dim_im_out_x, // output image dimension x or W
	const uint16_t dim_im_out_y, // output image dimension y or H
	q7_t *bufferA,               // NULL
	q7_t *Im_out);

void local_convolve_HWC_q7_nonsquare(const q7_t * Im_in,            // input image
	const uint16_t dim_im_in_x,  // input image dimention x
	const uint16_t dim_im_in_y,  // input image dimention y
	const uint16_t ch_im_in,     // number of input image channels
	const q7_t * wt,             // kernel weights 
	const uint16_t ch_im_out,    // number of filters, i.e., output image channels
	const uint16_t dim_kernel_x, // filter kernel size x
	const uint16_t dim_kernel_y, // filter kernel size y
	const uint16_t padding_x,    // padding sizes x
	const uint16_t padding_y,    // padding sizes y
	const uint16_t stride_x,     // stride x
	const uint16_t stride_y,     // stride y
	const q7_t * bias,           // bias
	const uint16_t bias_shift, const uint16_t out_shift, q7_t * Im_out,  // output image
	const uint16_t dim_im_out_x, // output image dimension x
	const uint16_t dim_im_out_y, // output image dimension y
	q15_t * bufferA,             //buffer space for input
	q7_t * bufferB);             //buffer space for output
									   
void local_convolve_CHW_q7_nonsquare(const q7_t * Im_in,            // input image
	const uint16_t dim_im_in_x,  // input image dimention x
	const uint16_t dim_im_in_y,  // input image dimention y
	const uint16_t ch_im_in,     // number of input image channels
	const q7_t * wt,             // kernel weights 
	const uint16_t ch_im_out,    // number of filters, i.e., output image channels
	const uint16_t dim_kernel_x, // filter kernel size x
	const uint16_t dim_kernel_y, // filter kernel size y
	const uint16_t padding_x,    // padding sizes x
	const uint16_t padding_y,    // padding sizes y
	const uint16_t stride_x,     // stride x
	const uint16_t stride_y,     // stride y
	const q7_t * bias,           // bias
	const uint16_t bias_shift, const uint16_t out_shift, q7_t * Im_out,  // output image
	const uint16_t dim_im_out_x, // output image dimension x
	const uint16_t dim_im_out_y, // output image dimension y
	q15_t * bufferA,             //buffer space for input
	q7_t * bufferB);             //buffer space for output

void local_depthwise_separable_conv_HWC_q7_nonsquare(const q7_t * Im_in,  // input image
	const uint16_t dim_im_in_x,  // input image dimention x
	const uint16_t dim_im_in_y,  // input image dimention y
	const uint16_t ch_im_in, // number of input image channels
	const q7_t * wt, // kernel weights 
	const uint16_t ch_im_out,    // number of filters, i.e., output image channels
	const uint16_t dim_kernel_x, // filter kernel size x
	const uint16_t dim_kernel_y, // filter kernel size y
	const uint16_t padding_x,    // padding sizes x
	const uint16_t padding_y,    // padding sizes y
	const uint16_t stride_x, // stride x
	const uint16_t stride_y, // stride y
	const q7_t * bias,   // bias
	const uint16_t bias_shift,   // amount of left-shift for bias
	const uint16_t out_shift,    // amount of right-shift for output
	q7_t * Im_out,   // output image
	const uint16_t dim_im_out_x, // output image dimension x
	const uint16_t dim_im_out_y, // output image dimension y
	q15_t * bufferA, //buffer space for input
	q7_t * bufferB);   //buffer space for output
													   
void local_depthwise_separable_conv_CHW_q7_nonsquare(const q7_t *Im_in,           // input image
	const uint16_t dim_im_in_x,  // input image dimention x
	const uint16_t dim_im_in_y,  // input image dimention y
	const uint16_t ch_im_in,     // number of input image channels
	const q7_t *wt,              // kernel weights
	const uint16_t ch_im_out,    // number of filters, i.e., output image channels
	const uint16_t dim_kernel_x, // filter kernel size x
	const uint16_t dim_kernel_y, // filter kernel size y
	const uint16_t padding_x,    // padding sizes x
	const uint16_t padding_y,    // padding sizes y
	const uint16_t stride_x,     // stride x
	const uint16_t stride_y,     // stride y
	const q7_t *bias,            // bias
	const uint16_t bias_shift,   // amount of left-shift for bias
	const uint16_t out_shift,    // amount of right-shift for output
	q7_t *Im_out,                // output image
	const uint16_t dim_im_out_x, // output image dimension x
	const uint16_t dim_im_out_y, // output image dimension y
	q15_t *bufferA,              //buffer space for input
	q7_t *bufferB);                //buffer space for output

void local_zero_padding_HWC_q7(const q7_t *Im_in,           // input image
	const uint16_t dim_im_in_x,    // input image dimention x
	const uint16_t dim_im_in_y,    // input image dimention y
	const uint16_t ch_im_in,       // number of input image channels
	const uint16_t padding_top,    // padding sizes y
	const uint16_t padding_bottom, // padding sizes y
	const uint16_t padding_left,   // padding sizes x
	const uint16_t padding_right,  // padding sizes x
	q7_t *Im_out,                  // output image
	const uint16_t dim_im_out_x,   // output image dimension x
	const uint16_t dim_im_out_y);  // output image dimension y 
						 
void local_zero_padding_CHW_q7(const q7_t *Im_in,           // input image
	const uint16_t dim_im_in_x,    // input image dimention x
	const uint16_t dim_im_in_y,    // input image dimention y
	const uint16_t ch_im_in,       // number of input image channels
	const uint16_t padding_top,    // padding sizes y
	const uint16_t padding_bottom, // padding sizes y
	const uint16_t padding_left,   // padding sizes x
	const uint16_t padding_right,  // padding sizes x
	q7_t *Im_out,                  // output image
	const uint16_t dim_im_out_x,   // output image dimension x
	const uint16_t dim_im_out_y);  // output image dimension y 
						 
void local_cropping_HWC_q7(const q7_t *Im_in,           // input image
	const uint16_t dim_im_in_x,    // input image dimention x
	const uint16_t dim_im_in_y,    // input image dimention y
	const uint16_t ch_im_in,       // number of input image channels
	const uint16_t padding_top,    // padding sizes y
	const uint16_t padding_bottom, // padding sizes y
	const uint16_t padding_left,   // padding sizes x
	const uint16_t padding_right,  // padding sizes x
	q7_t *Im_out,                  // output image
	const uint16_t dim_im_out_x,   // output image dimension x
	const uint16_t dim_im_out_y);  // output image dimension y 
						 
void local_cropping_CHW_q7(const q7_t *Im_in,           // input image
	const uint16_t dim_im_in_x,    // input image dimention x
	const uint16_t dim_im_in_y,    // input image dimention y
	const uint16_t ch_im_in,       // number of input image channels
	const uint16_t padding_top,    // padding sizes y
	const uint16_t padding_bottom, // padding sizes y
	const uint16_t padding_left,   // padding sizes x
	const uint16_t padding_right,  // padding sizes x
	q7_t *Im_out,                  // output image
	const uint16_t dim_im_out_x,   // output image dimension x
	const uint16_t dim_im_out_y);  // output image dimension y 

void local_fully_connected_q7_opt(const q7_t * pV,    // pointer to vector
	const q7_t * pM,    // pointer to matrix
	const uint16_t dim_vec, // length of the vector
	const uint16_t num_of_rows, // numCol of A
	const uint16_t bias_shift,  // amount of left-shift for bias
	const uint16_t out_shift,   // amount of right-shift for output
	const q7_t * bias, q7_t * pOut, // output operand
	q15_t * vec_buffer);


void local_fully_connected_q7(const q7_t * pV,    // pointer to vector
	const q7_t * pM,    // pointer to matrix
	const uint16_t dim_vec, // length of the vector
	const uint16_t num_of_rows, // numCol of A
	const uint16_t bias_shift,  // amount of left-shift for bias
	const uint16_t out_shift,   // amount of right-shift for output
	const q7_t * bias, q7_t * pOut, // output operand
	q15_t * vec_buffer);


// softmax
void local_softmax_q7(const q7_t * vec_in, const uint32_t dim_vec, q7_t * p_out);

// sigmoid
void local_sigmoid_q7(q7_t * data, uint32_t size, int16_t int_width);

// tanh
void local_tanh_q7(q7_t * data, uint32_t size, int16_t int_width);

// relu
void local_relu_q7(q7_t * data, uint32_t size);

// matrix ops
void local_mult_q7(q7_t * pSrcA, q7_t * pSrcB, q7_t * pDst, const uint16_t out_shift, uint32_t blockSize);

// add 
void local_add_q7(q7_t * pSrcA, q7_t * pSrcB, q7_t * pDst, const uint16_t out_shift,  uint32_t blockSize);

// sub 
void local_sub_q7(q7_t * pSrcA, q7_t * pSrcB, q7_t * pDst, const uint16_t out_shift, uint32_t blockSize);



// For more info. check CMSIS-NN lib
// https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/Source/NNSupportFunctions/arm_nntables.c
static const q7_t nnom_sigmoid_table_q7[256] = {
    0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e,
    0x50, 0x52, 0x53, 0x55, 0x57, 0x59, 0x5a, 0x5c,
    0x5e, 0x5f, 0x61, 0x62, 0x63, 0x65, 0x66, 0x67,
    0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70,
    0x71, 0x72, 0x72, 0x73, 0x74, 0x74, 0x75, 0x76,
    0x76, 0x77, 0x77, 0x78, 0x78, 0x79, 0x79, 0x7a,
    0x7a, 0x7a, 0x7b, 0x7b, 0x7b, 0x7c, 0x7c, 0x7c,
    0x7c, 0x7c, 0x7d, 0x7d, 0x7d, 0x7d, 0x7d, 0x7e,
    0x7e, 0x7e, 0x7e, 0x7e, 0x7e, 0x7e, 0x7e, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x05, 0x05, 0x05, 0x06,
    0x06, 0x06, 0x07, 0x07, 0x08, 0x08, 0x09, 0x09,
    0x0a, 0x0a, 0x0b, 0x0c, 0x0c, 0x0d, 0x0e, 0x0e,
    0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
    0x17, 0x19, 0x1a, 0x1b, 0x1d, 0x1e, 0x1f, 0x21,
    0x22, 0x24, 0x26, 0x27, 0x29, 0x2b, 0x2d, 0x2e,
    0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e,
};


static const q7_t nnom_tanh_table_q7[256] = {
    0x00, 0x08, 0x10, 0x18, 0x1f, 0x27, 0x2e, 0x35,
    0x3b, 0x41, 0x47, 0x4c, 0x51, 0x56, 0x5a, 0x5e,
    0x61, 0x65, 0x68, 0x6a, 0x6d, 0x6f, 0x71, 0x72,
    0x74, 0x75, 0x76, 0x78, 0x78, 0x79, 0x7a, 0x7b,
    0x7b, 0x7c, 0x7c, 0x7d, 0x7d, 0x7e, 0x7e, 0x7e,
    0x7e, 0x7e, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x81,
    0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0x82,
    0x82, 0x82, 0x82, 0x82, 0x83, 0x83, 0x84, 0x84,
    0x85, 0x85, 0x86, 0x87, 0x88, 0x88, 0x8a, 0x8b,
    0x8c, 0x8e, 0x8f, 0x91, 0x93, 0x96, 0x98, 0x9b,
    0x9f, 0xa2, 0xa6, 0xaa, 0xaf, 0xb4, 0xb9, 0xbf,
    0xc5, 0xcb, 0xd2, 0xd9, 0xe1, 0xe8, 0xf0, 0xf8,
};


#endif
