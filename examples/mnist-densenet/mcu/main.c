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
 
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "nnom.h"

#include "weights.h"

int8_t* load(const char* file)
{
	size_t sz;
	int8_t* in;
	FILE* fp = fopen(file,"rb");
	assert(fp);
	fseek(fp, 0, SEEK_END);
	sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	in = malloc(sz);
	fread(in, 1, sz, fp);
	fclose(fp);
	return in;
}

int main(int argc, char* argv[])
{
	nnom_model_t* model;
	int8_t* input;
	
	model = nnom_model_create();

	input = load("tmp/input.raw");

	memcpy(nnom_input_data, input, sizeof(nnom_input_data));
	model_run(model);

	model_delete(model);
	free(input);

	return 0;
}
