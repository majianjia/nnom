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
	nnom_predict_t * pre;
	int8_t* input;
	float prob;
	uint32_t label;
	size_t size = 0;

	input = load("test_data.bin", &size);	// load a continuous input dataset (test bin)
	fp = fopen("result.csv", "w");			// csv file for result
	fprintf(fp, "label, prob\n");				// header of csv
	printf("validation size: %d\n", (int)size); 
	
	model = nnom_model_create();				// create NNoM model
	pre = prediction_create(model, nnom_output_data, sizeof(nnom_output_data), 4); // mnist, 10 classes, get top-4
	
	// now takes label and data from the file and data
	for(size_t seek=0; seek < size;)
	{
		// labels
		uint8_t true_label[128];
		memcpy(true_label, input + seek, 128);
		seek += 128;
		// data
		for(int i=0; i < 128; i++)
		{
			if(seek >= size)
				break;
			memcpy(nnom_input_data, input + seek, sizeof(nnom_input_data));
			seek += sizeof(nnom_input_data);
			
			//nnom_predict(model, &label, &prob);				// this will work independently
			prediction_run(pre, true_label[i], &label, &prob);  // this provide more infor but requires prediction API
			
			// save results
			fprintf(fp, "%d,%f\n", label, prob);
		}
		printf("Processing %d%%\n", seek * 100 / size);
	}

	// print prediction result
	prediction_end(pre);
	prediction_summary(pre);
	prediction_delete(pre);

	// model
	model_stat(model);
	model_delete(model);

	fclose(fp);
	free(input);
	return 0;
}
