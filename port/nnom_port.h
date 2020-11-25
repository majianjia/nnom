/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-02-05     Jianjia Ma   The first version
 */

#ifndef __NNOM_PORT_H__
#define __NNOM_PORT_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

// use static memory 
#define NNOM_USING_STATIC_MEMORY    // enable to use built in memory allocation using large block of static memory
                                    // must set buf using "nnom_set_static_buf()" before creating a model. 

// dynamic memory interfaces
#ifndef NNOM_USING_STATIC_MEMORY    
    // when static memory is not used, you shall implement below memory interfaces (or use libc equivalent). 
    #define nnom_malloc(n)      malloc(n)       
    #define nnom_free(p)        free(p)
    #define nnom_memset(p,v,s)  memset(p,v,s)
#endif

// runtime & debug
#define nnom_us_get()       0       // return a microsecond timestamp
#define nnom_ms_get()       0       // return a millisecond timestamp
#define NNOM_LOG(...)       printf(__VA_ARGS__)

// NNoM configuration
#define NNOM_BLOCK_NUM  	(8)		// maximum number of memory blocks, increase it when log request.   
#define DENSE_WEIGHT_OPT 	(1)		// if used fully connected layer optimized weights. 

//#define NNOM_TRUNCATE             // disable: backend ops use round to the nearest int (default). enable: floor 

// Backend format configuration
//#define NNOM_USING_CHW            // uncomment if using CHW format. otherwise using default HWC format.
                                    // Notes, CHW is incompatible with CMSIS-NN. 
                                    // CHW must be used when using hardware accelerator such as KPU in K210 chip

// Backend selection
//#define NNOM_USING_CMSIS_NN       // uncomment if use CMSIS-NN for optimation 


#endif



