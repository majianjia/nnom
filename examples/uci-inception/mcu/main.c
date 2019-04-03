/*
 * Copyright (c) 2018-2019
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: LGPL-3.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-04-03     Jianjia Ma   The first version
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
#define DATA_TYPE_COUNT 	(6)		

nnom_model_t *model;

int main(void)
{
	model = nnom_model_create();
}


#ifdef RT_USING_FINSH
#include <finsh.h>
#include "math.h"
void nn_stat()
{
	model_stat(model);
	rt_kprintf("NNOM: Total Mem: %d\n", nnom_mem_stat());
}
MSH_CMD_EXPORT(nn_stat, print nn model);
FINSH_FUNCTION_EXPORT(nn_stat, nn_stat() to print data);

#endif


// ------- Using Y-modem to send test data set. -------------

#ifdef RT_USING_FINSH
#include <finsh.h>

#define DATA_SIZE (INPUT_CH * INPUT_WIDTH * INPUT_HIGHT)
#define LABEL_SIZE 128

static size_t file_total_size;

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
			rt_ringbuffer_get(ringbuffer, (uint8_t*)&nnom_input_data[0], DATA_SIZE);
			
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
	
	ringbuffer = rt_ringbuffer_create(4096);
	
	// delete first if it its not freed
	if(prediction!=NULL)
		predicetion_delete(prediction);
	// create new instance (test with all k)
	prediction = prediction_create(model, nnom_output_data, DATA_TYPE_COUNT, DATA_TYPE_COUNT-1);
	
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

#endif




