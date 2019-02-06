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


#ifndef __NNOM_UTILS_H__
#define __NNOM_UTILS_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"

typedef struct _nnom_predic_t
{
	uint16_t *confusion_mat;
	uint32_t *top_k;				// which, example: TOP2 num = top_k[0]+top_k[1]
	nnom_model_t * model;			// the model to run
	int8_t * buf_prediction;		// the pointer to the output of softmax layer(normally the end of classifier). 
	
	// setting
	uint32_t label_num;				// classification
	uint32_t top_k_size;			// number of k that wants to know. 
	
	// running
	uint32_t predic_count;			// how many prediction is done
	
	//timing
	uint32_t t_run_total; 			// total running time
	uint32_t t_predic_start;		// when it is initial
	uint32_t t_predic_total;		// total time of the whole test
} nnom_predic_t;


// create a prediction
// input model, the buf pointer to the softwmax output (Temporary, this can be extract from model)
// the size of softmax output (the num of lable)
// the top k that wants to record. 
nnom_predic_t* prediction_create(nnom_model_t* m, int8_t* buf_prediction, size_t label_num, size_t top_k_size);// currently int8_t 

// after a new data is set in input
// feed data to prediction
// input the current label, (range from 0 to total number of label -1)
// (the current input data should be set by user manully to the input buffer of the model.)
uint32_t prediction_run(nnom_predic_t* pre, uint32_t label);

// to mark prediction finished
void prediction_end(nnom_predic_t* pre);

// free all resources
void predicetion_delete(nnom_predic_t* pre);

// print matrix
void prediction_matrix(nnom_predic_t* pre);

// this function is to print sumarry 
void prediction_summary(nnom_predic_t* pre);


// -------------------------------

// stand alone prediction API
// this api test one set of data, return the prediction 
// input the model's input and output bufer
// return the predicted label
uint32_t nnom_predic_one(nnom_model_t* m, int8_t* input, int8_t* output); // currently int8_t 

void model_stat(nnom_model_t *m);

#endif


