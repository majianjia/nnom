/*
 * Copyright (c) 2018-2019, Jianjia Ma
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-03-29     Jianjia Ma   first implementation
 *
 * Notes:
 * This is a keyword spotting example using NNoM
 * 
 */

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "rtthread.h"

#include "nnom.h"
#include "weights.h"

#include "mfcc.h"
#include "stm32l4xx_hal.h"

// 
rt_event_t audio_evt;
rt_mutex_t mfcc_buf_mutex;

// NNoM model
nnom_model_t *model;

// 10 labels-1
//const char label_name[][10] =  {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknow"};

// 10 labels-2
//const char label_name[][10] =  {"marvin", "sheila", "yes", "no", "left", "right", "forward", "backward", "stop", "go", "unknow"};

// full 34 labels
const char label_name[][10] =  {"backward", "bed", "bird", "cat", "dog", "down", "eight","five", "follow", "forward",
                      "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right",
                      "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "yes", "zero", "unknow"};

// configuration
#define SAMP_FREQ 16000
#define AUDIO_FRAME_LEN (512) //31.25ms * 16000hz = 512, // FFT (windows size must be 2 power n)

mfcc_t * mfcc;
int32_t dma_audio_buffer[AUDIO_FRAME_LEN*2];
int16_t audio_buffer_16bit[(int)(AUDIO_FRAME_LEN*1.5)]; // an easy method for 50% overlapping


//the mfcc feature for kws
#define MFCC_LEN			(63)
#define MFCC_COEFFS_FIRST	(1)		// ignore the mfcc feature before this number
#define MFCC_COEFFS_LEN 	(13)    // the total coefficient to calculate
#define MFCC_COEFFS    	    (MFCC_COEFFS_LEN-MFCC_COEFFS_FIRST)

#define MFCC_FEAT_SIZE 	(MFCC_LEN * MFCC_COEFFS)
int8_t mfcc_features[MFCC_LEN][MFCC_COEFFS];	 // ring buffer
int8_t mfcc_features_seq[MFCC_LEN][MFCC_COEFFS]; // sequencial buffer for neural network input. 
uint32_t mfcc_feat_index = 0;

// msh debugging controls
bool is_print_abs_mean = false; // to print the mean of absolute value of the mfcc_features_seq[][]
bool is_print_mfcc  = false;    // to print the raw mfcc features at each update 
void Error_Handler()
{
	printf("error\n");
}

static TIM_HandleTypeDef s_TimerInstance = { 
    .Instance = TIM2
};
void us_timer_enable()
{
    __TIM2_CLK_ENABLE();
    s_TimerInstance.Init.Prescaler = 150;
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

DFSDM_Channel_HandleTypeDef  DfsdmChannelHandle;
DFSDM_Filter_HandleTypeDef   DfsdmFilterHandle;
DMA_HandleTypeDef            hDfsdmDma;

static void DFSDM_Init(void)
{
  /* Initialize channel 2 */
  __HAL_DFSDM_CHANNEL_RESET_HANDLE_STATE(&DfsdmChannelHandle);
  DfsdmChannelHandle.Instance                      = DFSDM1_Channel2;
  DfsdmChannelHandle.Init.OutputClock.Activation   = ENABLE;
  DfsdmChannelHandle.Init.OutputClock.Selection    = DFSDM_CHANNEL_OUTPUT_CLOCK_AUDIO;
  DfsdmChannelHandle.Init.OutputClock.Divider      = 16; /* 24.578MHz/16 = 1.536MHz */
  DfsdmChannelHandle.Init.Input.Multiplexer        = DFSDM_CHANNEL_EXTERNAL_INPUTS;
  DfsdmChannelHandle.Init.Input.DataPacking        = DFSDM_CHANNEL_STANDARD_MODE; /* N.U. */
  DfsdmChannelHandle.Init.Input.Pins               = DFSDM_CHANNEL_SAME_CHANNEL_PINS;
  DfsdmChannelHandle.Init.SerialInterface.Type     = DFSDM_CHANNEL_SPI_RISING;
  DfsdmChannelHandle.Init.SerialInterface.SpiClock = DFSDM_CHANNEL_SPI_CLOCK_INTERNAL;
  DfsdmChannelHandle.Init.Awd.FilterOrder          = DFSDM_CHANNEL_FASTSINC_ORDER; /* N.U. */
  DfsdmChannelHandle.Init.Awd.Oversampling         = 10; /* N.U. */
  DfsdmChannelHandle.Init.Offset                   = 0;
  DfsdmChannelHandle.Init.RightBitShift            = 2;
  if(HAL_OK != HAL_DFSDM_ChannelInit(&DfsdmChannelHandle))
  {
    Error_Handler();
  }

  /* Initialize filter 0 */
  __HAL_DFSDM_FILTER_RESET_HANDLE_STATE(&DfsdmFilterHandle);
  DfsdmFilterHandle.Instance                          = DFSDM1_Filter0;
  DfsdmFilterHandle.Init.RegularParam.Trigger         = DFSDM_FILTER_SW_TRIGGER;
  DfsdmFilterHandle.Init.RegularParam.FastMode        = ENABLE;
  DfsdmFilterHandle.Init.RegularParam.DmaMode         = ENABLE;
  DfsdmFilterHandle.Init.InjectedParam.Trigger        = DFSDM_FILTER_SW_TRIGGER; /* N.U. */
  DfsdmFilterHandle.Init.InjectedParam.ScanMode       = ENABLE; /* N.U. */
  DfsdmFilterHandle.Init.InjectedParam.DmaMode        = DISABLE; /* N.U. */
  DfsdmFilterHandle.Init.InjectedParam.ExtTrigger     = DFSDM_FILTER_EXT_TRIG_TIM1_TRGO; /* N.U. */
  DfsdmFilterHandle.Init.InjectedParam.ExtTriggerEdge = DFSDM_FILTER_EXT_TRIG_RISING_EDGE; /* N.U. */
  DfsdmFilterHandle.Init.FilterParam.SincOrder        = DFSDM_FILTER_SINC3_ORDER;
  DfsdmFilterHandle.Init.FilterParam.Oversampling     = 96; /* 11.294MHz/(4*64) = 44.1KHz */ // 1.536M/96 = 16k
  DfsdmFilterHandle.Init.FilterParam.IntOversampling  = 1;
  if(HAL_OK != HAL_DFSDM_FilterInit(&DfsdmFilterHandle))
  {
    Error_Handler();
  }

  /* Configure regular channel and continuous mode for filter 0 */
  if(HAL_OK != HAL_DFSDM_FilterConfigRegChannel(&DfsdmFilterHandle, DFSDM_CHANNEL_2, DFSDM_CONTINUOUS_CONV_ON))
  {
    Error_Handler();
  }
}



void microphone_init(void )
{	
	DFSDM_Init();
	HAL_DFSDM_FilterRegularStart_DMA(&DfsdmFilterHandle, dma_audio_buffer, 1024);
}

static int32_t abs_mean(int8_t *p, size_t size)
{
	int64_t sum = 0;
	for(size_t i = 0; i<size; i++)
	{
		if(p[i] < 0)
			sum+=-p[i];
		else
			sum += p[i];
	}
	return sum/size;
}

void thread_kws_serv(void *p)
{
	#define SaturaLH(N, L, H) (((N)<(L))?(L):(((N)>(H))?(H):(N)))
	uint32_t evt;
	int32_t *p_raw_audio;
	uint32_t time;
	
	// calculate 13 coefficient, use number #2~13 coefficient. discard #1
	mfcc = mfcc_create(MFCC_COEFFS_LEN, MFCC_COEFFS_FIRST, AUDIO_FRAME_LEN, 5, 0.97f); 

	while(1)
	{
		// wait for event and check which buffer is filled
		rt_event_recv(audio_evt, 1|2, RT_EVENT_FLAG_OR | RT_EVENT_FLAG_CLEAR, RT_WAITING_FOREVER, &evt);
		
		if(evt & 1)
			p_raw_audio = dma_audio_buffer;
		else
			p_raw_audio = &dma_audio_buffer[AUDIO_FRAME_LEN];
		
		// memory move
		// audio buffer = | 256 byte old data |   256 byte new data 1 | 256 byte new data 2 | 
		//                         ^------------------------------------------|
		memcpy(audio_buffer_16bit, &audio_buffer_16bit[AUDIO_FRAME_LEN], (AUDIO_FRAME_LEN/2)*sizeof(int16_t));
		
		// convert it to 16 bit. 
		// volume*4
		for(int i = 0; i < AUDIO_FRAME_LEN; i++)
		{
			audio_buffer_16bit[AUDIO_FRAME_LEN/2+i] = SaturaLH((p_raw_audio[i] >> 8)*1, -32768, 32767);
		}
		
		// MFCC
		// do the first mfcc with half old data(256) and half new data(256)
		// then do the second mfcc with all new data(512). 
		// take mfcc buffer
		
		rt_mutex_take(mfcc_buf_mutex, RT_WAITING_FOREVER);
		for(int i=0; i<2; i++)
		{
			mfcc_compute(mfcc, &audio_buffer_16bit[i*AUDIO_FRAME_LEN/2], mfcc_features[mfcc_feat_index]);
			
			// debug only, to print mfcc data on console
			if(is_print_mfcc)
			{
				for(int i=0; i<MFCC_COEFFS; i++)
					printf("%d ",  mfcc_features[mfcc_feat_index][i]);
				printf("\n");
			}
			
			mfcc_feat_index++;
			if(mfcc_feat_index >= MFCC_LEN)
				mfcc_feat_index = 0;
		}
		
		// release mfcc buffer
		rt_mutex_release(mfcc_buf_mutex);
	}
}



int main(void)
{
	uint32_t last_mfcc_index = 0; 
	uint32_t label;
	rt_tick_t last_time = 0;
	float prob;
	uint8_t priority = RT_THREAD_PRIORITY_MAX-2;
	
	us_timer_enable();
	
	// create thread sync 
	mfcc_buf_mutex = rt_mutex_create("mfcc_buf", RT_IPC_FLAG_FIFO);
	audio_evt = rt_event_create("evt_kws", RT_IPC_FLAG_FIFO);
	
	// create and compile the model 
	model = nnom_model_create();
	
	// change to lowest priority, avoid blocking shell
	rt_thread_control(rt_thread_self(), RT_THREAD_CTRL_CHANGE_PRIORITY, &priority);
	
	// create kws workers
	rt_thread_startup(rt_thread_create("kws_serv", thread_kws_serv, RT_NULL, 1024, 5, 50));
	
	// inite microphone
	microphone_init();
	
	while(1)
	{
		// mfcc wait for new data, then copy
		while(last_mfcc_index == mfcc_feat_index)
			rt_thread_delay(1);
		
		// copy mfcc ring buffer to sequance buffer. 
		rt_mutex_take(mfcc_buf_mutex, RT_WAITING_FOREVER);
		last_mfcc_index = mfcc_feat_index;
		uint32_t len_first = MFCC_FEAT_SIZE - mfcc_feat_index * MFCC_COEFFS;
		uint32_t len_second = mfcc_feat_index * MFCC_COEFFS;
		memcpy(&mfcc_features_seq[0][0], &mfcc_features[0][0] + len_second,  len_first);
		memcpy(&mfcc_features_seq[0][0] + len_first, &mfcc_features[0][0], len_second);
		rt_mutex_release(mfcc_buf_mutex);
		
		// debug only, to print the abs mean of mfcc output. use to adjust the dec bit (shifting)
		// of the mfcc computing. 
		if(is_print_abs_mean)
			printf("abs mean:%d\n", abs_mean((int8_t*)mfcc_features_seq, MFCC_FEAT_SIZE));
		
		// ML
		memcpy(nnom_input_data, mfcc_features_seq, MFCC_FEAT_SIZE);
		nnom_predict(model, &label, &prob);
		
		// output
		if(prob > 0.5f)
		{
			last_time = rt_tick_get();
			printf("%s : %d%%\n", (char*)&label_name[label], (int)(prob * 100));
		}
	}
}

// half callback
void HAL_DFSDM_FilterRegConvHalfCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter)
{
	// make sure audio event is initialized 
	if(audio_evt)
		rt_event_send(audio_evt, 1);
}
// full callback
void HAL_DFSDM_FilterRegConvCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter)
{
	if(audio_evt)
		rt_event_send(audio_evt, 2);
}


void HAL_DFSDM_ChannelMspInit(DFSDM_Channel_HandleTypeDef *hdfsdm_channel)
{
  /* Init of clock, gpio and PLLSAI1 clock */
  GPIO_InitTypeDef GPIO_Init;
  RCC_PeriphCLKInitTypeDef RCC_PeriphCLKInitStruct;
  
  /* Enable DFSDM clock */
  __HAL_RCC_DFSDM1_CLK_ENABLE();
  
  /* Configure PE9 for DFSDM_CKOUT and PE7 for DFSDM_DATIN2 */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  GPIO_Init.Mode      = GPIO_MODE_AF_PP;
  GPIO_Init.Pull      = GPIO_PULLDOWN;
  GPIO_Init.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_Init.Alternate = GPIO_AF6_DFSDM1;
  GPIO_Init.Pin = GPIO_PIN_9;
  HAL_GPIO_Init(GPIOE, &GPIO_Init);
  GPIO_Init.Pin = GPIO_PIN_7;
  HAL_GPIO_Init(GPIOE, &GPIO_Init);
  
  /* Configure and enable PLLSAI1 clock to generate 11.294MHz */
  RCC_PeriphCLKInitStruct.PeriphClockSelection    = RCC_PERIPHCLK_SAI1;
  RCC_PeriphCLKInitStruct.PLLSAI1.PLLSAI1Source   = RCC_PLLSOURCE_MSI;
  RCC_PeriphCLKInitStruct.PLLSAI1.PLLSAI1M        = 1;
  RCC_PeriphCLKInitStruct.PLLSAI1.PLLSAI1N        = 43;
  RCC_PeriphCLKInitStruct.PLLSAI1.PLLSAI1P        = 7;//
  RCC_PeriphCLKInitStruct.PLLSAI1.PLLSAI1ClockOut = RCC_PLLSAI1_SAI1CLK;
  RCC_PeriphCLKInitStruct.Sai1ClockSelection      = RCC_SAI1CLKSOURCE_PLLSAI1;
  if(HAL_RCCEx_PeriphCLKConfig(&RCC_PeriphCLKInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

void HAL_DFSDM_FilterMspInit(DFSDM_Filter_HandleTypeDef *hdfsdm_filter)
{
  /* Configure DMA1_Channel4 */
  __HAL_RCC_DMA1_CLK_ENABLE();
  hDfsdmDma.Init.Request             = DMA_REQUEST_0;
  hDfsdmDma.Init.Direction           = DMA_PERIPH_TO_MEMORY;
  hDfsdmDma.Init.PeriphInc           = DMA_PINC_DISABLE;
  hDfsdmDma.Init.MemInc              = DMA_MINC_ENABLE;
  hDfsdmDma.Init.PeriphDataAlignment = DMA_PDATAALIGN_WORD;
  hDfsdmDma.Init.MemDataAlignment    = DMA_MDATAALIGN_WORD;
  hDfsdmDma.Init.Mode                = DMA_CIRCULAR;
  hDfsdmDma.Init.Priority            = DMA_PRIORITY_HIGH;
  hDfsdmDma.Instance                 = DMA1_Channel4;
  __HAL_LINKDMA(hdfsdm_filter, hdmaReg, hDfsdmDma);
  if (HAL_OK != HAL_DMA_Init(&hDfsdmDma))
  {
    Error_Handler();
  }
  HAL_NVIC_SetPriority(DMA1_Channel4_IRQn, 0x01, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel4_IRQn);
}

void DMA1_Channel4_IRQHandler(void)
{
	HAL_DMA_IRQHandler(&hDfsdmDma);
}


// Msh functions

#ifdef RT_USING_FINSH
#include <finsh.h>
void nn_stat()
{
	model_stat(model);
	rt_kprintf("Total Memory cost (Network and NNoM): %d\n", nnom_mem_stat());
}
MSH_CMD_EXPORT(nn_stat, print nn model);

void kws_mfcc()
{
	is_print_mfcc = !is_print_mfcc;
}
MSH_CMD_EXPORT(kws_mfcc, print the raw mfcc values);

void kws_mean()
{
	is_print_abs_mean = !is_print_abs_mean;
}
MSH_CMD_EXPORT(kws_mean, print the abs mean value of mfcc output);

#endif















