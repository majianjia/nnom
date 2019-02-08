/*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2018-05-14     ZYH          first implementation
 */
 
#include <stdio.h>
#include "stm32l4xx_hal.h"
#include "rtthread.h"
#include "rtdevice.h"
#include "nnom.h"

#include "weights.h"
#include "ymodem.h"

#include "nnom.h"

// STM32 TIMER
static TIM_HandleTypeDef s_TimerInstance = { 
    .Instance = TIM2
};
void us_timer_enable()
{
    __TIM2_CLK_ENABLE();
    s_TimerInstance.Init.Prescaler = 80;
    s_TimerInstance.Init.CounterMode = TIM_COUNTERMODE_UP;
    s_TimerInstance.Init.Period = 0xffffffff;
    s_TimerInstance.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    s_TimerInstance.Init.RepetitionCounter = 0;
    HAL_TIM_Base_Init(&s_TimerInstance);
    HAL_TIM_Base_Start(&s_TimerInstance);
}
uint32_t us_timer_get()
{
	return __HAL_TIM_GET_COUNTER(&s_TimerInstance);
}


// input data (int8 or q7)
#define INPUT_RATE			50
#define INPUT_CH			9	 
#define INPUT_WIDTH			128
#define INPUT_HIGHT			1
#define DATA_TYPE_COUNT 	(6)		


// weights and bias ----------------
const int8_t conv1_wt[] = CONV1D_1_KERNEL_0;
const int8_t conv1_b[] = CONV1D_1_BIAS_0;
const int8_t conv2_wt[] = CONV1D_2_KERNEL_0;
const int8_t conv2_b[] = CONV1D_2_BIAS_0;
const int8_t conv3_wt[] = CONV1D_3_KERNEL_0;
const int8_t conv3_b[] = CONV1D_3_BIAS_0;
const int8_t conv4_wt[] = CONV1D_4_KERNEL_0;
const int8_t conv4_b[] = CONV1D_4_BIAS_0;
const int8_t fc1_wt[] = DENSE_1_KERNEL_0;
const int8_t fc1_b[] = DENSE_1_BIAS_0;
const int8_t fc2_wt[] = DENSE_2_KERNEL_0;
const int8_t fc2_b[] = DENSE_2_BIAS_0;

nnom_weight_t c1_w = {
	.p_value = (void*)conv1_wt,
	.shift = CONV1D_1_KERNEL_0_SHIFT};

nnom_bias_t c1_b = {
	.p_value = (void*)conv1_b,
	.shift = CONV1D_1_BIAS_0_SHIFT};

nnom_weight_t c2_w = {
	.p_value = (void*)conv2_wt,
	.shift = CONV1D_2_KERNEL_0_SHIFT};

nnom_bias_t c2_b = {
	.p_value = (void*)conv2_b,
	.shift = CONV1D_2_BIAS_0_SHIFT};

nnom_weight_t c3_w = {
	.p_value = (void*)conv3_wt,
	.shift = CONV1D_3_KERNEL_0_SHIFT};

nnom_bias_t c3_b = {
	.p_value = (void*)conv3_b,
	.shift = CONV1D_3_BIAS_0_SHIFT};

nnom_weight_t c4_w = {
	.p_value = (void*)conv4_wt,
	.shift = CONV1D_4_KERNEL_0_SHIFT};

nnom_bias_t c4_b = {
	.p_value = (void*)conv4_b,
	.shift = CONV1D_4_BIAS_0_SHIFT};

nnom_weight_t ip1_w = {
	.p_value = (void*)fc1_wt,
	.shift = DENSE_1_KERNEL_0_SHIFT};

nnom_bias_t ip1_b = {
	.p_value = (void*)fc1_b,
	.shift = DENSE_1_BIAS_0_SHIFT};

nnom_weight_t ip2_w = {
	.p_value = (void*)fc2_wt,
	.shift = DENSE_2_KERNEL_0_SHIFT};

nnom_bias_t ip2_b = {
	.p_value = (void*)fc2_b,
	.shift = DENSE_2_BIAS_0_SHIFT};

// a global model for used in console
nnom_model_t model = {0}; 

// input output buffer. 
int8_t nnom_input_data[INPUT_HIGHT * INPUT_WIDTH * INPUT_CH];
int8_t nnom_output_data[DATA_TYPE_COUNT];

int main(void)
{
	nnom_layer_t *input_layer;
	nnom_layer_t *x;
	nnom_layer_t *x1;		
	nnom_layer_t *x2;
	nnom_layer_t *x3;
	
	// for runtime stat
	us_timer_enable();

	// inital a model
	new_model(&model);
	
	// input layer
	input_layer = Input(shape(INPUT_HIGHT, INPUT_WIDTH, INPUT_CH), qformat(7, 0), nnom_input_data);
	
	// conv2d
	x = model.hook(Conv2D(16, kernel(1, 9), stride(1, 2), PADDING_SAME, &c1_w, &c1_b), input_layer);
	x = model.active(act_relu(), x);
	x = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x);
	
	// conv2d - 1 - inception
	x1 = model.hook(Conv2D(16, kernel(1, 5), stride(1, 1), PADDING_SAME, &c2_w, &c2_b), x);
	x1 = model.active(act_relu(), x1);
	x1 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x1);
	
	// conv2d - 2 - inception
	x2 = model.hook(Conv2D(16, kernel(1, 3), stride(1, 1), PADDING_SAME, &c3_w, &c3_b), x);
	x2 = model.active(act_relu(), x2);
	x2 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x2);
	
	// maxpool - 3 - inception
	x3 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x);
	
	// concatenate 
	x = model.merge(Concat(-1), x1, x2); 
	x = model.merge(Concat(-1), x, x3);
	
	// conv2d conclusion of inception 
	x = model.hook(Conv2D(48, kernel(1, 3), stride(1, 1), PADDING_SAME, &c4_w, &c4_b), x);
	x = model.active(act_relu(), x);
	x = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x);
		
	// flatten & dense
	x = model.hook(Flatten(), x);
	x = model.hook(Dense(128, &ip1_w, &ip1_b), x);
	x = model.active(act_relu(), x);
	x = model.hook(Dense(6, &ip2_w, &ip2_b), x);
	x = model.hook(Softmax(), x);
	
	// output layer
	x = model.hook(Output(shape(6,1,1), qformat(7, 0), nnom_output_data), x);
	
	// compile and check
	model_compile(&model, input_layer, x);
	
	// run once
	model_run(&model);
	
	// the prediction will be in console, so we do nothing in main. 
	while(1)
	{
		rt_thread_delay(RT_TICK_PER_SECOND);
	}

}


#ifdef RT_USING_FINSH
#include <finsh.h>
#include "math.h"
void nn_stat()
{
	model_stat(&model);
	rt_kprintf("NNOM: Total Mem: %d\n", nnom_mem_stat());
}
MSH_CMD_EXPORT(nn_stat, print nn model);
FINSH_FUNCTION_EXPORT(nn_stat, nn_stat() to print data);

#endif


// test -------------------------- Using Y-modem to send test data set. 

#ifdef RT_USING_FINSH
#include <finsh.h>

#define DATA_SIZE (INPUT_CH * INPUT_WIDTH * INPUT_HIGHT)
#define LABEL_SIZE 128

static size_t file_total_size, file_cur_size;

//test
struct rt_ringbuffer*  ringbuffer = RT_NULL;
nnom_predic_t *prediction = NULL;

// parameters
uint8_t	 test_label[LABEL_SIZE] = {0};  // where a batch of label stores
uint32_t test_label_countdown = 0;		// count down of that batch
uint32_t test_total_count = 0;			// 

static enum rym_code ymodem_on_begin(struct rym_ctx *ctx, rt_uint8_t *buf, rt_size_t len) {
	char *file_name, *file_size;

	/* calculate and store file size */
	file_name = (char *) &buf[0];
	file_size = (char *) &buf[rt_strlen(file_name) + 1];
	file_total_size = atol(file_size);
	/* 4 bytes align */
	file_total_size = (file_total_size + 3) / 4 * 4;
	file_cur_size = 0;
	
	// local data size
	test_label_countdown = 0;
	test_total_count = 0;
	memset(test_label, 0, LABEL_SIZE);
	
	return RYM_CODE_ACK;
}

static enum rym_code ymodem_on_data(struct rym_ctx *ctx, rt_uint8_t *buf, rt_size_t len) 
{
	// put data in buffer, then get it as block. 
	rt_ringbuffer_put(ringbuffer, buf, len);
	
	while(1)
	{
		// get label. 
		if(test_label_countdown == 0 && rt_ringbuffer_data_len(ringbuffer) >= LABEL_SIZE)
		{
			// get the label, reset the label countdown. 	
			rt_ringbuffer_get(ringbuffer, &test_label[0], LABEL_SIZE);
			test_label_countdown = LABEL_SIZE;
		}
		
		// if there is enough data and the label is still availble. 
		if(test_label_countdown > 0 && rt_ringbuffer_data_len(ringbuffer) >= DATA_SIZE)
		{
			// use one lata
			test_label_countdown --;
			
			// get input data
			rt_ringbuffer_get(ringbuffer, &nnom_input_data[0], DATA_SIZE);
			
			// do this prediction round.
			prediction_run(prediction, test_label[test_total_count % LABEL_SIZE]);
			
			// we can use the count in prediction as well.
			test_total_count += 1;
		}
		// return while there isnt enough data
		else
		{
			return RYM_CODE_ACK;
		}
	}
}


void predic() 
{
	struct rym_ctx rctx;

	rt_kprintf("Please select the NNoM binary test file and use Ymodem-128/1024  to send.\n");

	// preparing for prediction 
	us_timer_enable();
	
	ringbuffer = rt_ringbuffer_create(4096);
	
	// delete first if it its not freed
	if(prediction!=NULL)
		predicetion_delete(prediction);
	// create new instance (test with all k)
	prediction = prediction_create(&model, nnom_output_data, DATA_TYPE_COUNT, DATA_TYPE_COUNT-1);
	
	// begin
	// data is feed in receiving callback
	if (!rym_recv_on_device(&rctx, rt_console_get_device(), RT_DEVICE_OFLAG_RDWR | RT_DEVICE_FLAG_INT_RX,
			ymodem_on_begin, ymodem_on_data, NULL, RT_TICK_PER_SECOND)) {
		/* wait some time for terminal response finish */
		rt_thread_delay(RT_TICK_PER_SECOND / 10);
		rt_kprintf("\nPrediction done.\n");

	} else {
		/* wait some time for terminal response finish */
		rt_thread_delay(RT_TICK_PER_SECOND / 10);
		rt_kprintf("Test file incompleted. \n Partial results are shown below:\n");
	}
	// finished
	prediction_end(prediction);
	// print sumarry & matrix
	prediction_summary(prediction);
	
	// free buffer
	rt_ringbuffer_destroy(ringbuffer);
	// predicetion_delete(prediction); // optional to free data now


}
FINSH_FUNCTION_EXPORT(predic, validate NNoM model implementation with test set);
MSH_CMD_EXPORT(predic, validate NNoM model implementation with test set);

void matrix()
{
	if(prediction != NULL)
		prediction_matrix(prediction);
}
FINSH_FUNCTION_EXPORT(matrix, matrix() to print confusion matrix);
MSH_CMD_EXPORT(matrix, print confusion matrix);

void reboot() 
{
	printf("\nThe answer is...%f\n", 42.0f);
	rt_thread_delay(RT_TICK_PER_SECOND);
	NVIC_SystemReset();
}

FINSH_FUNCTION_EXPORT(reboot, reboot() );
MSH_CMD_EXPORT(reboot, reboot system);
#endif




