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
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "nnom.h"

#include "weights.h"

#define INPUT_CH			1	 
#define INPUT_WIDTH			28
#define INPUT_HIGHT			28
#define NUM_CLASS 			(10)		


// NN ----------------
const int8_t conv1_wt[] = CONV2D_1_KERNEL_0;
const int8_t conv1_b[] = CONV2D_1_BIAS_0;
const int8_t conv2_wt[] = CONV2D_2_KERNEL_0;
const int8_t conv2_b[] = CONV2D_2_BIAS_0;
const int8_t conv3_wt[] = CONV2D_3_KERNEL_0;
const int8_t conv3_b[] = CONV2D_3_BIAS_0;
const int8_t conv4_wt[] = CONV2D_4_KERNEL_0;
const int8_t conv4_b[] = CONV2D_4_BIAS_0;
const int8_t conv5_wt[] = CONV2D_5_KERNEL_0;
const int8_t conv5_b[] = CONV2D_5_BIAS_0;
//const int8_t conv6_wt[] = CONV2D_6_KERNEL_0;
//const int8_t conv6_b[] = CONV2D_6_BIAS_0;
//const int8_t conv7_wt[] = CONV2D_7_KERNEL_0;
//const int8_t conv7_b[] = CONV2D_7_BIAS_0;
//const int8_t conv8_wt[] = CONV2D_8_KERNEL_0;
//const int8_t conv8_b[] = CONV2D_8_BIAS_0;
//const int8_t conv9_wt[] = CONV2D_9_KERNEL_0;
//const int8_t conv9_b[] = CONV2D_9_BIAS_0;
//const int8_t conv10_wt[] = CONV2D_10_KERNEL_0;
//const int8_t conv10_b[] = CONV2D_10_BIAS_0;
//const int8_t conv11_wt[] = CONV2D_11_KERNEL_0;
//const int8_t conv11_b[] = CONV2D_11_BIAS_0;
const int8_t fc1_wt[] = DENSE_1_KERNEL_0;
const int8_t fc1_b[] = DENSE_1_BIAS_0;
//const int8_t fc2_wt[] = DENSE_2_KERNEL_0;
//const int8_t fc2_b[] = DENSE_2_BIAS_0;

nnom_weight_t c1_w = {
	.p_value = (void*)conv1_wt,
	.shift = CONV2D_1_KERNEL_0_SHIFT};

nnom_bias_t c1_b = {
	.p_value = (void*)conv1_b,
	.shift =  CONV2D_1_BIAS_LSHIFT};

nnom_weight_t c2_w = {
	.p_value = (void*)conv2_wt,
	.shift = CONV2D_2_OUTPUT_RSHIFT};

nnom_bias_t c2_b = {
	.p_value = (void*)conv2_b,
	.shift = CONV2D_2_BIAS_LSHIFT};

nnom_weight_t c3_w = {
	.p_value = (void*)conv3_wt,
	.shift = CONV2D_3_OUTPUT_RSHIFT};

nnom_bias_t c3_b = {
	.p_value = (void*)conv3_b,
	.shift = CONV2D_3_BIAS_LSHIFT};

nnom_weight_t c4_w = {
	.p_value = (void*)conv4_wt,
	.shift = CONV2D_4_OUTPUT_RSHIFT};

nnom_bias_t c4_b = {
	.p_value = (void*)conv4_b,
	.shift = CONV2D_4_BIAS_LSHIFT};

nnom_weight_t c5_w = {
	.p_value = (void*)conv5_wt,
	.shift = CONV2D_5_OUTPUT_RSHIFT};

nnom_bias_t c5_b = {
	.p_value = (void*)conv5_b,
	.shift = CONV2D_5_BIAS_LSHIFT};

//nnom_weight_t c6_w = {
//	.p_value = (void*)conv6_wt,
//	.shift = CONV2D_6_OUTPUT_RSHIFT};

//nnom_bias_t c6_b = {
//	.p_value = (void*)conv6_b,
//	.shift = CONV2D_6_BIAS_LSHIFT};

//nnom_weight_t c7_w = {
//	.p_value = (void*)conv7_wt,
//	.shift = CONV2D_7_OUTPUT_RSHIFT};

//nnom_bias_t c7_b = {
//	.p_value = (void*)conv7_b,
//	.shift = CONV2D_7_BIAS_LSHIFT};

//nnom_weight_t c8_w = {
//	.p_value = (void*)conv8_wt,
//	.shift = CONV2D_8_OUTPUT_RSHIFT};

//nnom_bias_t c8_b = {
//	.p_value = (void*)conv8_b,
//	.shift = CONV2D_8_BIAS_LSHIFT};

//nnom_weight_t c9_w = {
//	.p_value = (void*)conv9_wt,
//	.shift = CONV2D_9_OUTPUT_RSHIFT};

//nnom_bias_t c9_b = {
//	.p_value = (void*)conv9_b,
//	.shift = CONV2D_9_BIAS_LSHIFT};

//nnom_weight_t c10_w = {
//	.p_value = (void*)conv10_wt,
//	.shift = CONV2D_10_OUTPUT_RSHIFT};

//nnom_bias_t c10_b = {
//	.p_value = (void*)conv10_b,
//	.shift = CONV2D_10_BIAS_LSHIFT};

//nnom_weight_t c11_w = {
//	.p_value = (void*)conv11_wt,
//	.shift = CONV2D_11_OUTPUT_RSHIFT};

//nnom_bias_t c11_b = {
//	.p_value = (void*)conv11_b,
//	.shift = CONV2D_11_BIAS_LSHIFT};

nnom_weight_t ip1_w = {
	.p_value = (void*)fc1_wt,
	.shift = DENSE_1_OUTPUT_RSHIFT};

nnom_bias_t ip1_b = {
	.p_value = (void*)fc1_b,
	.shift = DENSE_1_BIAS_LSHIFT};

//nnom_weight_t ip2_w = {
//	.p_value = (void*)fc2_wt,
//	.shift = DENSE_2_OUTPUT_RSHIFT};

//nnom_bias_t ip2_b = {
//	.p_value = (void*)fc2_b,
//	.shift = DENSE_2_BIAS_LSHIFT};

nnom_model_t model = {0}; // to use finsh to print
int8_t nnom_input_data[INPUT_HIGHT * INPUT_WIDTH * INPUT_CH];
int8_t nnom_output_data[NUM_CLASS];

nnom_layer_t * dense_block1(nnom_model_t* model, nnom_layer_t * in, uint32_t k)
{
	nnom_layer_t * x[5];
	
	x[0] = in;
	
	x[1] = model->hook(Conv2D(k, kernel(3, 3), stride(1, 1), PADDING_SAME, &c2_w, &c2_b), x[0]);
	x[1] = model->active(act_relu(), x[1]);

	x[2] = model->mergex(Concat(-1), 2, x[0], x[1]);
	x[2] = model->hook(Conv2D(k, kernel(3, 3), stride(1, 1), PADDING_SAME, &c3_w, &c3_b), x[2]);
	x[2] = model->active(act_relu(), x[2]);
	
	x[3] = model->mergex(Concat(-1), 3, x[0], x[1], x[2]);
	x[3] = model->hook(Conv2D(k, kernel(3, 3), stride(1, 1), PADDING_SAME, &c4_w, &c4_b), x[3]);
	x[3] = model->active(act_relu(), x[3]);
	
	x[4] = model->mergex(Concat(-1), 4, x[0], x[1], x[2], x[3]);
	x[4] = model->hook(Conv2D(k, kernel(3, 3), stride(1, 1), PADDING_SAME, &c5_w, &c5_b), x[4]);
	x[4] = model->active(act_relu(), x[4]);
	
	return model->mergex(Concat(-1), 5, x[0], x[1], x[2], x[3], x[4]);
}

//nnom_layer_t * dense_block2(nnom_model_t* model, nnom_layer_t * in, uint32_t k)
//{
//	nnom_layer_t * x[5];
//	
//	x[0] = in;
//	
//	x[1] = model->hook(Conv2D(k, kernel(3, 3), stride(1, 1), PADDING_SAME, &c7_w, &c7_b), x[0]);
//	x[1] = model->active(act_relu(), x[1]);

//	x[2] = model->mergex(Concat(-1), 2, x[0], x[1]);
//	x[2] = model->hook(Conv2D(k, kernel(3, 3), stride(1, 1), PADDING_SAME, &c8_w, &c8_b), x[2]);
//	x[2] = model->active(act_relu(), x[2]);
//	
//	x[3] = model->mergex(Concat(-1), 3, x[0], x[1], x[2]);
//	x[3] = model->hook(Conv2D(k, kernel(3, 3), stride(1, 1), PADDING_SAME, &c9_w, &c9_b), x[3]);
//	x[3] = model->active(act_relu(), x[3]);
//	
//	x[4] = model->mergex(Concat(-1), 4, x[0], x[1], x[2], x[3]);
//	x[4] = model->hook(Conv2D(k, kernel(3, 3), stride(1, 1), PADDING_SAME, &c10_w, &c10_b), x[4]);
//	x[4] = model->active(act_relu(), x[4]);
//	
//	return model->mergex(Concat(-1), 5, x[0], x[1], x[2], x[3], x[4]);
//}

int8_t* load(const char* file)
{
	size_t sz;
	int8_t* in;
	FILE* fp = fopen(file,"rb");
	assert(fp);
	fseek(fp, 0, SEEK_END);
	sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	in = malloc(sz);
	fread(in, 1, sz, fp);
	fclose(fp);
	return in;
}

int main(int argc, char* argv[])
{
	nnom_layer_t *input_layer;
	nnom_layer_t *x;
	int8_t* input;
	
	// inital a model
	new_model(&model);
	
	// input format
	input_layer = Input(shape(INPUT_HIGHT, INPUT_WIDTH, INPUT_CH), qformat(7, 0), nnom_input_data);
	
	// first conv
	x = model.hook(Conv2D(8, kernel(7, 7), stride(1, 1), PADDING_SAME, &c1_w, &c1_b), input_layer);
	x = model.active(act_relu(), x);
	x = model.hook(MaxPool(kernel(4, 4), stride(4, 4), PADDING_SAME), x);
	
	// dense block 1, 
	x = dense_block1(&model, x, 24);
	
	// bottleneck 
//	x = model.hook(Conv2D(32, kernel(1, 1), stride(1, 1), PADDING_SAME, &c6_w, &c6_b), x);
//	x = model.active(act_relu(), x);
//	x = model.hook(MaxPool(kernel(2, 2), stride(2, 2), PADDING_SAME), x);
//	
//	// dense block 2, growth rate k = 12
//	//x = dense_block2(&model, x, 12);
//	
	// reduce channel for global average
//	x = model.hook(Conv2D(10, kernel(1, 1), stride(1, 1), PADDING_SAME, &c7_w, &c7_b), x);
//	x = model.active(act_relu(), x);
	
	// global average pooling
	x = model.hook(GlobalMaxPool(), x);
	//x = model.hook(GlobalAvgPool(), x);
	//x = model.hook(AvgPool(kernel(14, 14), stride(7, 7), PADDING_VALID), x);
	//x = model.hook(MaxPool(kernel(7, 7), stride(7, 7), PADDING_VALID), x);
		
	// flatten & dense
	x = model.hook(Flatten(), x);
	x = model.hook(Dense(10, &ip1_w, &ip1_b), x);
	
//	x = model.hook(Dense(128, &ip1_w, &ip1_b), x);
//	x = model.active(act_relu(), x);
//	x = model.hook(Dense(NUM_CLASS, &ip2_w, &ip2_b), x);
	
	
	x = model.hook(Softmax(), x);
	x = model.hook(Output(shape(NUM_CLASS, 1, 1), qformat(7, 0), nnom_output_data), x);
	
	// compile and check
	model_compile(&model, input_layer, x);

	input = load("tmp/input.raw");

	memcpy(nnom_input_data, input, INPUT_HIGHT*INPUT_WIDTH*INPUT_CH);
	model_run(&model);

    free(input);

	return 0;
}
