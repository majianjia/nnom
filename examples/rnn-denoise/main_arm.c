/*
 * Copyright (c) 2018-2020, Jianjia Ma
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-09-09     Jianjia Ma   The first version
 *
 *
 * This file is apart of NNoM examples, which aims to provide clear demo of every steps. 
 * Therefor, it is not optimized for neither space and speed. 
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "nnom.h"
#include "weights.h"
#include "mfcc.h"

 // the bandpass filter coefficiences
#include "equalizer_coeff.h" 

#define NUM_FEATURES NUM_FILTER

#define _MAX(x, y) (((x) > (y)) ? (x) : (y))
#define _MIN(x, y) (((x) < (y)) ? (x) : (y))

#define NUM_CHANNELS 	1
#define SAMPLE_RATE 	16000
#define AUDIO_FRAME_LEN 512

// audio buffer for input
float audio_buffer[AUDIO_FRAME_LEN] = {0};
int16_t audio_buffer_16bit[AUDIO_FRAME_LEN] = {0};

// buffer for output
int16_t audio_buffer_filtered[AUDIO_FRAME_LEN/2] = { 0 };


// mfcc features and their derivatives
float mfcc_feature[NUM_FEATURES] = { 0 };
float mfcc_feature_prev[NUM_FEATURES] = { 0 };
float mfcc_feature_diff[NUM_FEATURES] = { 0 };
float mfcc_feature_diff_prev[NUM_FEATURES] = { 0 };
float mfcc_feature_diff1[NUM_FEATURES] = { 0 };
// features for NN
float nn_features[64] = {0};
int8_t nn_features_q7[64] = {0};

// NN results, which is the gains for each frequency band
float band_gains[NUM_FILTER] = {0};
float band_gains_prev[NUM_FILTER] = {0};

// 0db gains coefficient
float coeff_b[NUM_FILTER][NUM_COEFF_PAIR] = FILTER_COEFF_B;
float coeff_a[NUM_FILTER][NUM_COEFF_PAIR] = FILTER_COEFF_A;
// dynamic gains coefficient
float b_[NUM_FILTER][NUM_COEFF_PAIR] = {0};

// nnom model
nnom_model_t *model;

// for microphone related data. 
volatile bool is_half_updated = false;
volatile bool is_full_updated = false;
int32_t dma_audio_buffer[AUDIO_FRAME_LEN]; 

// update the history
void y_h_update(float *y_h, uint32_t len)
{
	for (uint32_t i = len-1; i >0 ;i--)
		y_h[i] = y_h[i-1];
}

//  equalizer by multiple n order iir band pass filter. 
// y[i] = b[0] * x[i] + b[1] * x[i - 1] + b[2] * x[i - 2] - a[1] * y[i - 1] - a[2] * y[i - 2]...
void equalizer(float* x, float* y, uint32_t signal_len, float *b, float *a, uint32_t num_band, uint32_t num_order)
{
	// the y history for each band
	static float y_h[NUM_FILTER][NUM_COEFF_PAIR] = { 0 };
	static float x_h[NUM_COEFF_PAIR * 2] = { 0 };
	uint32_t num_coeff = num_order * 2 + 1;

	// i <= num_coeff (where historical x is involved in the first few points)
	// combine state and new data to get a continual x input. 
	memcpy(x_h + num_coeff, x, num_coeff * sizeof(float));
	for (uint32_t i = 0; i < num_coeff; i++)
	{
		y[i] = 0;
		for (uint32_t n = 0; n < num_band; n++)
		{
			y_h_update(y_h[n], num_coeff);
			y_h[n][0] = b[n * num_coeff] * x_h[i+ num_coeff];
			for (uint32_t c = 1; c < num_coeff; c++)
				y_h[n][0] += b[n * num_coeff + c] * x_h[num_coeff + i - c] - a[n * num_coeff + c] * y_h[n][c];
			y[i] += y_h[n][0];
		}
	}
	// store the x for the state of next round
	memcpy(x_h, &x[signal_len - num_coeff], num_coeff * sizeof(float));
	
	// i > num_coeff; the rest data not involed the x history
	for (uint32_t i = num_coeff; i < signal_len; i++)
	{
		y[i] = 0;
		for (uint32_t n = 0; n < num_band; n++)
		{
			y_h_update(y_h[n], num_coeff);
			y_h[n][0] = b[n * num_coeff] * x[i];
			for (uint32_t c = 1; c < num_coeff; c++)
				y_h[n][0] += b[n * num_coeff + c] * x[i - c] - a[n * num_coeff + c] * y_h[n][c];
			y[i] += y_h[n][0];
		}	
	}
}

// set dynamic gains. Multiple gains x b_coeff
void set_gains(float *b_in, float *b_out,  float* gains, uint32_t num_band, uint32_t num_order)
{
	uint32_t num_coeff = num_order * 2 + 1;
	for (uint32_t i = 0; i < num_band; i++)
		for (uint32_t c = 0; c < num_coeff; c++)
			b_out[num_coeff *i + c] = b_in[num_coeff * i + c] * gains[i]; 
}

void quantize_data(float*din, int8_t *dout, uint32_t size, uint32_t int_bit)
{
	float limit = (1 << int_bit); 
	for(uint32_t i=0; i<size; i++)
		dout[i] = (int8_t)(_MAX(_MIN(din[i], limit), -limit) / limit * 127);
}

int main(int argc, char* argv[])
{
	#define SaturaLH(N, L, H) (((N)<(L))?(L):(((N)>(H))?(H):(N)))
	void us_timer_enable();
	void microphone_init();
	void LED_Init();
	void led(bool flag);
	uint32_t us_timer_get();
	
	// batchmarking time
	uint32_t start_time = 0;
	uint32_t mfcc_time = 0;
	uint32_t nn_time = 0;
	uint32_t equalizer_time = 0;
	
	int32_t *p_new_data;
	
	// NNoM model
	model = nnom_model_create();
	
	// mfcc features, 0 offset, 26 bands, 512fft, 0 preempha, attached_energy_to_band0
	mfcc_t * mfcc = mfcc_create(NUM_FEATURES, 0, NUM_FEATURES, 512, 0, true);
	
	// microphone and timer, and LED
	us_timer_enable();
	microphone_init();
	LED_Init();

	while(1) {
		// move buffer (50%) overlapping, move later 50% to the first 50, then fill 
		memcpy(audio_buffer_16bit, &audio_buffer_16bit[AUDIO_FRAME_LEN/2], AUDIO_FRAME_LEN/2*sizeof(int16_t));
		
		// banchmarking
		//printf("mfcc:%dus, nn:%dus, EQ:%dus, total:%dus\n", mfcc_time, nn_time, equalizer_time, us_timer_get()-start_time);
		
		// wait for new data
		while(!is_half_updated && !is_full_updated);
		// record start time
		start_time = us_timer_get();
		
		if(is_half_updated){
			is_half_updated = false;
			p_new_data = dma_audio_buffer;
		}
		else
		{
			is_full_updated = false;
			p_new_data = &dma_audio_buffer[AUDIO_FRAME_LEN/2];
		}
		// convert the file to 16bit.
		for(int i = 0; i < AUDIO_FRAME_LEN/2; i++)
			audio_buffer_16bit[AUDIO_FRAME_LEN/2+i] = SaturaLH((p_new_data[i] >> 8), -32768, 32767);
		
		// get mfcc
		mfcc_compute(mfcc, audio_buffer_16bit, mfcc_feature);
		mfcc_time = us_timer_get() - start_time;
		
		// get the first and second derivative of mfcc
		for(uint32_t i=0; i< NUM_FEATURES; i++)
		{
			mfcc_feature_diff[i] = mfcc_feature[i] - mfcc_feature_prev[i];
			mfcc_feature_diff1[i] = mfcc_feature_diff[i] - mfcc_feature_diff_prev[i];
		}
		memcpy(mfcc_feature_prev, mfcc_feature, NUM_FEATURES * sizeof(float));
		memcpy(mfcc_feature_diff_prev, mfcc_feature_diff, NUM_FEATURES * sizeof(float));
		
		// combine MFCC with derivatives for the NN features
		memcpy(nn_features, mfcc_feature, NUM_FEATURES*sizeof(float));
		memcpy(&nn_features[NUM_FEATURES], mfcc_feature_diff, 10*sizeof(float));
		memcpy(&nn_features[NUM_FEATURES+10], mfcc_feature_diff1, 10*sizeof(float));

		// quantise them using the same scale as training data (in keras), by 2^n. 
		quantize_data(nn_features, nn_features_q7, NUM_FEATURES+20, 3);
		
		// run the mode with the new input
		memcpy(nnom_input_data, nn_features_q7, sizeof(nnom_input_data));
		model_run(model);
		nn_time = us_timer_get() - start_time - mfcc_time;
		
		// read the result, convert it back to float (q0.7 to float)
		for(int i=0; i< NUM_FEATURES; i++)
			band_gains[i] = (float)(nnom_output_data[i]) / 127.f;
		
		// one more step, limit the change of gians, to smooth the speech, per RNNoise paper
		for(int i=0; i< NUM_FEATURES; i++)
			band_gains[i] = _MAX(band_gains_prev[i]*0.8f, band_gains[i]); 
		memcpy(band_gains_prev, band_gains, NUM_FEATURES *sizeof(float));
		
		// update filter coefficient to applied dynamic gains to each frequency band 
		set_gains((float*)coeff_b, (float*)b_, band_gains, NUM_FILTER, NUM_ORDER);

		// convert 16bit to float for equalizer
		for (int i = 0; i < AUDIO_FRAME_LEN/2; i++)
			audio_buffer[i] = audio_buffer_16bit[i + AUDIO_FRAME_LEN / 2] / 32768.f;
				
		// finally, we apply the equalizer to this audio frame to denoise
		equalizer(audio_buffer, &audio_buffer[AUDIO_FRAME_LEN / 2], AUDIO_FRAME_LEN/2, (float*)b_,(float*)coeff_a, NUM_FILTER, NUM_ORDER);
		equalizer_time = us_timer_get() - start_time - (mfcc_time + nn_time);
		
		// convert it back to int16
		for (int i = 0; i < AUDIO_FRAME_LEN / 2; i++)
			audio_buffer_filtered[i] = audio_buffer[i + AUDIO_FRAME_LEN / 2] * 32768.f *0.6f; // 0.7 is the filter band overlapping factor
		
		// voice detection to show an LED PE8 -> GreenLED
		if(nnom_output_data1[0] >= 64)
			led(true);
		else
			led(false);
		
		// filtered frame is now in audio_buffer_filtered[], size = half_frame
		// you may implement your own method to store or playback the voice. 
	}

}



// microphone implementation. 
#include "stm32l4xx_hal.h"

static void LED_Init(void)         
{
  GPIO_InitTypeDef GPIO_InitStruct;

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();

  /*Configure GPIO pin : Led_Pin */
  GPIO_InitStruct.Pin = GPIO_PIN_8;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;   
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
}
void led(bool flag)
{
	if(flag)
		HAL_GPIO_WritePin(GPIOE, GPIO_PIN_8, GPIO_PIN_SET);
	else
		HAL_GPIO_WritePin(GPIOE, GPIO_PIN_8, GPIO_PIN_RESET);
}

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
    s_TimerInstance.Init.Prescaler = HAL_RCC_GetSysClockFreq()/1000000;
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
  DfsdmChannelHandle.Init.RightBitShift            = 2;//2;
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
  DfsdmFilterHandle.Init.FilterParam.SincOrder        = DFSDM_FILTER_SINC3_ORDER; // DFSDM_FILTER_SINC3_ORDER = 19bit valide data (+-262144)
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
	HAL_DFSDM_FilterRegularStart_DMA(&DfsdmFilterHandle, dma_audio_buffer, AUDIO_FRAME_LEN);
}


// half callback
void HAL_DFSDM_FilterRegConvHalfCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter)
{
	is_half_updated = true;
}
// full callback
void HAL_DFSDM_FilterRegConvCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter)
{
	is_full_updated = true;
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
  
  /* Configure and enable PLLSAI1 clock to generate 24.57148MHz */
  RCC_PeriphCLKInitStruct.PeriphClockSelection    = RCC_PERIPHCLK_SAI1;
  RCC_PeriphCLKInitStruct.PLLSAI1.PLLSAI1Source   = RCC_PLLSOURCE_MSI;
  RCC_PeriphCLKInitStruct.PLLSAI1.PLLSAI1M        = 1;
  RCC_PeriphCLKInitStruct.PLLSAI1.PLLSAI1N        = 43;
  RCC_PeriphCLKInitStruct.PLLSAI1.PLLSAI1P        = 7;// 24.5714
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




