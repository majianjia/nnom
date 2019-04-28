#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "nnom.h"
#include "weights_cnn.h"


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
nnom_status_t layer_callback(nnom_model_t *m, nnom_layer_t *layer)
{
	static int outputIndex[NNOM_TYPE_MAX] = { 0 , } ;
	char name[32];
	FILE* fp;

	outputIndex[layer->type]++;
	snprintf(name, sizeof(name),"tmp/%s%d.raw",
			default_layer_names[layer->type],
			outputIndex[layer->type]);
	fp = fopen(name,"w");
	if(fp != NULL)
	{
		fwrite(layer->out->mem->blk, 1, shape_size(&layer->out->shape), fp);
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

	model = nnom_model_create();
	model_set_callback(model, layer_callback); // callback to save output of each layer.
	
	input = load("tmp/input.raw");

	memcpy(nnom_input_data, input, sizeof(nnom_input_data));
	model_run(model);

	model_delete(model);
	free(input);
	return 0;
}
