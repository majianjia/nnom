#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "weights_cnn.h"


#include "nnom.h"

q7_t* load(const char* file)
{
	size_t sz;
	q7_t* in;
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

#define CONV1_IM_DIM 28
#define CONV1_IM_CH  1
#define CONV1_OUT_CH 32
#define CONV1_KER_DIM 5
#define CONV1_PADDING ((CONV1_KER_DIM-1)/2)
#define CONV1_STRIDE  1
#define CONV1_OUT_DIM 28

#define POOL1_KER_DIM 2
#define POOL1_PADDING ((POOL1_KER_DIM-1)/2)
#define POOL1_STRIDE  2
#define POOL1_OUT_DIM 14

#define CONV2_IM_DIM 14
#define CONV2_IM_CH  32
#define CONV2_OUT_DIM 14
#define CONV2_OUT_CH 64
#define CONV2_STRIDE 1
#define CONV2_KER_DIM 5
#define CONV2_PADDING ((CONV2_KER_DIM-1)/2)

#define POOL2_KER_DIM 2
#define POOL2_PADDING ((POOL2_KER_DIM-1)/2)
#define POOL2_STRIDE  2
#define POOL2_OUT_DIM 7

#define IP1_DIM 3136
#define IP1_OUT 1024

#define IP2_DIM 1024
#define IP2_OUT 10


q7_t y_out[10];

static const q7_t W_conv1[] = CONV2D_1_KERNEL_0;
static const q7_t B_conv1[] = CONV2D_1_BIAS_0;
static const q7_t W_conv2[] = CONV2D_2_KERNEL_0;
static const q7_t B_conv2[] = CONV2D_2_BIAS_0;
static const q7_t W_fc1[] = DENSE_1_KERNEL_0;
static const q7_t B_fc1[] = DENSE_1_BIAS_0;
static const q7_t W_fc2[] = DENSE_2_KERNEL_0;
static const q7_t B_fc2[] = DENSE_2_BIAS_0;

static const nnom_weight_t c1_w = {W_conv1, CONV2D_1_OUTPUT_RSHIFT};
static const nnom_bias_t   c1_b = {B_conv1, CONV2D_1_BIAS_LSHIFT};
static const nnom_weight_t c2_w = {W_conv2, CONV2D_2_OUTPUT_RSHIFT};
static const nnom_bias_t   c2_b = {B_conv2, CONV2D_2_BIAS_LSHIFT};
static const nnom_weight_t ip1_w = {W_fc1, DENSE_1_OUTPUT_RSHIFT};
static const nnom_bias_t   ip1_b = {B_fc1, DENSE_1_BIAS_LSHIFT};
static const nnom_weight_t ip2_w = {W_fc2, DENSE_2_OUTPUT_RSHIFT};
static const nnom_bias_t   ip2_b = {B_fc2, DENSE_2_BIAS_LSHIFT};

int main(int argc, char* argv[])
{
	q7_t *input;
	printf("loading input&weights...\n");
	input = load("tmp/input.raw");
	nnom_model_t model;
	new_model(&model);
	model.add(&model, Input(shape(CONV1_IM_DIM, CONV1_IM_DIM, CONV1_IM_CH), qformat(0, 7), input));
	model.add(&model, Conv2D(CONV1_OUT_CH, kernel(CONV1_KER_DIM, CONV1_KER_DIM), stride(CONV1_STRIDE, CONV1_STRIDE), PADDING_SAME, &c1_w, &c1_b));
	model.add(&model, ReLU());
	model.add(&model, MaxPool(kernel(POOL1_KER_DIM, POOL1_KER_DIM), stride(POOL1_STRIDE, POOL1_STRIDE), PADDING_VALID));
	model.add(&model, Conv2D(CONV2_OUT_CH, kernel(CONV2_KER_DIM, CONV2_KER_DIM), stride(CONV2_STRIDE, CONV2_STRIDE), PADDING_SAME, &c2_w, &c2_b));
	model.add(&model, ReLU());
	model.add(&model, MaxPool(kernel(POOL2_KER_DIM, POOL2_KER_DIM), stride(POOL2_STRIDE, POOL2_STRIDE), PADDING_VALID));
	model.add(&model, Dense(IP1_OUT, &ip1_w, &ip1_b));
	model.add(&model, ReLU());
	model.add(&model, Dense(IP2_OUT, &ip2_w, &ip2_b));
	model.add(&model, Softmax());
	model.add(&model, Output(shape(IP2_OUT, 1, 1), qformat(0, 7), y_out));
	sequencial_compile(&model);

	model_run(&model);

	model_delete(&model);
	printf("inference is done!\n");

	free(input);
	return 0;
}
