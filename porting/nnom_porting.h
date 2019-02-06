/*
 * Copyright (c) 2018-2019
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * This file (nnom_porting.h) is UNLICENSED, independently to NNoM. 
 * 
 *	
 * Change Logs:
 * Date           Author       Notes
 * 2019-02-05     Jianjia Ma   The first version
 */

#ifndef __NNOM_PORTING_H__
#define __NNOM_PORTING_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

// memory interfaces
#define nnom_malloc(n)   	malloc(n) 
#define nnom_free(p)		free(p)
#define nnom_memset(p,v,s)  memset(p,v,s)

// runtime & debuges
#define nnom_us_get()		0
#define nnom_ms_get()		0
#define LOG(...)			printf(__VA_ARGS__)

// NNoM configuration
#define NNOM_BLOCK_NUM  	(8)		// maximum number of memory block  
#define DENSE_WEIGHT_OPT 	(1)		// if used fully connected layer optimized weights. 


// An porting example shows below
/*

#include "rtthread.h"
extern uint32_t us_timer_get(void);

// memory interfaces
#define nnom_malloc(n)   	rt_malloc(n) 
#define nnom_free(p)		rt_free(p)
#define nnom_memset(p,v,s)  rt_memset(p,v,s)

// runtime & debuges
#define nnom_us_get()		us_timer_get()
#define nnom_ms_get()		rt_tick_get()
#define LOG(...)			rt_kprintf(__VA_ARGS__)

// NNoM configuration
#define NNOM_BLOCK_NUM  	(8)		
#define DENSE_WEIGHT_OPT 	(1)		
*/

#endif



