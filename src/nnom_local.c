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

#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "nnom_local.h"

/**
   * @brief Q7 max pooling function
   * @param[in, out]  Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimention
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   Im_out      pointer to output tensor
   * @return none.
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * bufferA size:  0
   *
   * The pooling function is implemented as split x-pooling then
   * y-pooling.
   *
   * This pooling function is input-destructive. Input data is undefined
   * after calling this function.
   *
   */

void arm_maxpool_1d_q7_HWC(q7_t *Im_in,
                           const uint16_t dim_im_in,
                           const uint16_t ch_im_in,
                           const uint16_t dim_kernel,
                           const uint16_t padding,
                           const uint16_t stride, const uint16_t dim_im_out, q7_t *bufferA, q7_t *Im_out)
{

  /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

  int16_t i_ch_in, i_x; // input
  int16_t k_x;          // output

  // channel (depth)
  for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
  {
    // x
    for (i_x = 0; i_x < dim_im_out; i_x++)
    {
      //pooling
      int max = -129;

      for (k_x = i_x * stride - padding; k_x < i_x * stride - padding + dim_kernel; k_x++)
      {
        if (k_x >= 0 && k_x < dim_im_in)
        {
          if (Im_in[i_ch_in + ch_im_in * k_x] > max)
          {
            max = Im_in[i_ch_in + ch_im_in * k_x];
          }
        }
      }

      Im_out[i_ch_in + ch_im_in * i_x] = max;
    }
  }
}

/**
   * @brief Q7 average pooling function
   * @param[in,out]   Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimention
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   Im_out      pointer to output tensor
   * @return none.
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * bufferA size:  2*dim_im_out*ch_im_in // in this implementation, no buffer is needed. 
   *
   * The pooling function is implemented as split x-pooling then
   * y-pooling.
   *
   * This pooling function is input-destructive. Input data is undefined
   * after calling this function.
   *
   */

void arm_avepool_1d_q7_HWC(q7_t *Im_in,
                           const uint16_t dim_im_in,
                           const uint16_t ch_im_in,
                           const uint16_t dim_kernel,
                           const uint16_t padding,
                           const uint16_t stride, const uint16_t dim_im_out, q7_t *bufferA, q7_t *Im_out)
{

  int16_t i_ch_in, i_x;
  int16_t k_x;

  for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
  {
    // x
    for (i_x = 0; i_x < dim_im_out; i_x++)
    {
      // pooling
      int sum = 0;
      int count = 0;

      for (k_x = i_x * stride - padding; k_x < i_x * stride - padding + dim_kernel; k_x++)
      {
        if (k_x >= 0 && k_x < dim_im_in)
        {
          sum += Im_in[i_ch_in + ch_im_in * k_x];
          count++;
        }
      }

      Im_out[i_ch_in + ch_im_in * i_x] = sum / count;
    }
  }
}
