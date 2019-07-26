/*
 * Copyright (c) 2018-2019, Jianjia Ma
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-07-25     Jianjia Ma   The first version
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

nnom_status_t layer_callback(nnom_model_t *m, nnom_layer_t *layer)
{
	static int outputIndex[NNOM_TYPE_MAX] = { 0 , } ;
	static int layer_index = 0;
	char name[32];
	FILE* fp;

	outputIndex[layer->type]++;
	snprintf(name, sizeof(name),"tmp/nnom_%d_%s%d.raw",
			(layer_index++),
			default_layer_names[layer->type],
			outputIndex[layer->type]);
	fp = fopen(name,"w");
	if(fp != NULL)
	{
		fwrite(layer->out->tensor->p_data, 1, tensor_size(layer->out->tensor), fp);
		fclose(fp);
	}
	else
	{
		printf("failed to save %s\n",name);
	}
	return NN_SUCCESS;
}


int main(int argc, char* argv[])
{
	nnom_model_t* model;
	int8_t* input;
	size_t size = 0;

	input = load("tmp/input.raw", &size);
	model = nnom_model_create();	
	model_set_callback(model, layer_callback); // callback to save output of each layer.	
	
	memcpy(nnom_input_data, input, sizeof(nnom_input_data));
	model_run(model);
	model_delete(model);
	
	free(input);
	return 0;
}
