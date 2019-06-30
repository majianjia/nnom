/*
 * Copyright (c) 2018-2019, Jianjia Ma
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-06-30     Jianjia Ma   The first version
 */
 
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "nnom.h"

#include "weights.h"

int8_t* load(const char* file, size_t * size)
{
	size_t sz;
	FILE* fp = fopen(file,"rb");
	int8_t* input;
	assert(fp);
	fseek(fp, 0, SEEK_END);
	sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	input = malloc(sz);
	fread(input, 1, sz, fp);
	fclose(fp);
	*size = sz;
	return input;
}

int main(int argc, char* argv[])
{
	FILE* fp;
	nnom_model_t* model;
	//nnom_predict_t * pre;
	int8_t* input;
	float prob;
	uint32_t label;
	size_t size = 0;
	
	model = nnom_model_create();			// create NNoM model
	input = load("tmp/input.raw", &size);	// load a continuous input dataset
	printf("validation size: %d\n", size);	
	fp = fopen("tmp/result.csv", "w");		// file for result
	fprintf(fp, "label, prob\n");
	
	//pre = prediction_create(model, nnom_input_data, 10, 9);
	
	// do inference for each input data, the input size is same as input buffer define in "weights.h"
	for(long i=0; i*sizeof(nnom_input_data) < size; i++)
	{
		memcpy(nnom_input_data, input + i*sizeof(nnom_input_data), sizeof(nnom_input_data));
		nnom_predict(model, &label, &prob);
		//label = prediction_run(pre, 0); prob = 1;
		fprintf(fp, "%d,%f\n", label, prob);
	}
	prediction_summary(pre);
	fclose(fp);
	model_delete(model);
	free(input);
	
	return 0;
}
