/*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: LGPL-3.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-03-29     Jianjia Ma   first implementation
 */
 
#include <stdio.h>
#include "rtthread.h"

#include "nnom.h"
#include "image.h"
#include "weights.h"

nnom_model_t *model;

int main(void)
{
	rt_thread_delay(10);

	// create and compile the model 
	model = nnom_model_create();
	
	// dummy run
	model_run(model);
}


#ifdef RT_USING_FINSH
#include <finsh.h>
// ASCII lib from (https://www.jianshu.com/p/1f58a0ebf5d9)
const char codeLib[] = "@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.   ";
void print_img(int8_t * buf)
{
    for(int y = 0; y < 28; y++) 
	{
        for (int x = 0; x < 28; x++) 
		{
            int index =  69 / 127.0 * (127 - buf[y*28+x]); 
			if(index > 69) index =69;
			if(index < 0) index = 0;
            printf("%c",codeLib[index]);
			printf("%c",codeLib[index]);
        }
        printf("\n");
    }
}

// Do simple test using image in "image.h" with model created previously. 
void mnist(int argc, char** argv)
{
	uint32_t tick, time;
	int32_t result;
	int32_t index = atoi(argv[1]);
	
	if(index >= TOTAL_IMAGE || argc != 2)
	{
		printf("Please input image number within %d\n", TOTAL_IMAGE-1);
		return;
	}
	
	printf("\nprediction start.. \n");
	tick = rt_tick_get();
	
	// copy data and do prediction
	memcpy(nnom_input_data, (int8_t*)&img[index][0], 784);
	result = nnom_predic_one(model);
	time = rt_tick_get() - tick;
	
	//print original image to console
	print_img((int8_t*)&img[index][0]);
	
	printf("Time: %d tick\n", time);
	printf("Truth label: %d\n", label[index]);
	printf("Predicted label: %d\n", result);
}

FINSH_FUNCTION_EXPORT(mnist, mnist(4) );
MSH_CMD_EXPORT(mnist, mnist);

void nn_stat()
{
	model_stat(model);
	rt_kprintf("Total Memory cost (Network and NNoM): %d\n", nnom_mem_stat());
}
MSH_CMD_EXPORT(nn_stat, print nn model);
FINSH_FUNCTION_EXPORT(nn_stat, nn_stat() to print data);
#endif
