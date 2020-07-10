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

// input data (int8 or q7)
#define INPUT_RATE			50
#define INPUT_CH			9	 
#define INPUT_WIDTH			128
#define INPUT_HIGHT			1
#define NUM_CLASS 			(6)		

nnom_model_t *model;

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

int main(void)
{
	rt_thread_delay(10);
	
	// for runtime stat
	us_timer_enable();

	model = nnom_model_create();
	
	// run once
	model_run(model);
}


#ifdef RT_USING_FINSH
#include <finsh.h>
#include "math.h"
void nn_stat()
{
	model_stat(model);
	rt_kprintf("Total Memory cost (Network and NNoM): %d\n", nnom_mem_stat());
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
nnom_predict_t *prediction = NULL;

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
	float prob;
	uint32_t label;
	
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
			rt_ringbuffer_get(ringbuffer, (uint8_t*)nnom_input_data, DATA_SIZE);
			
			// do this prediction round.
			prediction_run(prediction, test_label[test_total_count % LABEL_SIZE], &label, &prob);  
			
			
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
void predict() 
{
	struct rym_ctx rctx;

	rt_kprintf("Please select the NNoM binary test file and use Ymodem-128/1024  to send.\n");

	// preparing for prediction 
	us_timer_enable();
	
	ringbuffer = rt_ringbuffer_create(4096);
	
	// delete first if it its not freed
	if(prediction!=NULL)
		prediction_delete(prediction);
	// create new instance (test with all k)
	prediction = prediction_create(model, nnom_output_data, NUM_CLASS, NUM_CLASS-1);
	
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
	// prediction_delete(prediction); // optional to free data now


}
FINSH_FUNCTION_EXPORT(predict, validate NNoM model implementation with test set);
MSH_CMD_EXPORT(predict, validate NNoM model implementation with test set);

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

// memory test
void create_model() 
{
	printf("Model created \n");
	main();
}

FINSH_FUNCTION_EXPORT(create_model, create_model  );
MSH_CMD_EXPORT(create_model, create_model );

// memory test
void delete_model() 
{
	model_delete(model);
	printf("Model deleted \n");
}

FINSH_FUNCTION_EXPORT(delete_model, delete_model model );
MSH_CMD_EXPORT(delete_model, delete_model model);
#endif





