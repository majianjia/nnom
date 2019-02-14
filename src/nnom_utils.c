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
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "nnom.h"
#include "nnom_utils.h"

static nnom_predic_t *_predic_create_instance(nnom_model_t *m, size_t label_num, size_t top_k_size)
{
	nnom_predic_t *pre;
	uint8_t *p;
	size_t mem_size = 0;
	mem_size += alignto(label_num * label_num * 2, 4); // confusion_mat
	mem_size += top_k_size * 4;						   // top_k
	mem_size += alignto(sizeof(nnom_predic_t), 4);

	// we dont use nnom_mem(), we dont count the memory
	p = nnom_malloc(mem_size);
	if (!p)
		return NULL;
	nnom_memset(p, 0, mem_size);

	pre = (nnom_predic_t *)p;
	pre->confusion_mat = (uint16_t *)(p + alignto(sizeof(nnom_predic_t), 4));
	pre->top_k = (uint32_t *)(p + alignto(sizeof(nnom_predic_t), 4) + alignto(label_num * label_num * 2, 4));

	// config
	pre->label_num = label_num;
	pre->top_k_size = top_k_size;
	pre->predic_count = 0;

	// run
	pre->model = m;

	pre->t_run_total = 0;	// model running time in total
	pre->t_predic_start = 0; // when it is initial
	pre->t_predic_total = 0; // total time of the whole test

	return pre;
}

// create a prediction
// input model, the buf pointer to the softwmax output (Temporary, this can be extract from model)
// the size of softmax output (the num of lable)
// the top k that wants to record.
nnom_predic_t *prediction_create(nnom_model_t *m, int8_t *buf_prediction, size_t label_num, size_t top_k_size)
{
	nnom_predic_t *pre = _predic_create_instance(m, label_num, top_k_size);
	if (!pre)
		return NULL;
	if (!m)
	{
		nnom_free(pre);
		return NULL;
	}

	// set the output buffer of model to the prediction instance
	pre->buf_prediction = buf_prediction;

	// mark start time.
	pre->t_predic_start = nnom_ms_get();

	return pre;
}

// after a new data is set in input
// feed data to prediction
// input the current label, (range from 0 to total number of label -1)
// (the current input data should be set by user manully to the input buffer of the model.)
int32_t prediction_run(nnom_predic_t *pre, uint32_t label)
{
	int max_val;
	int max_index;
	uint32_t true_ranking = 0;
	uint32_t start;

	if (!pre)
		return NN_ARGUMENT_ERROR;

	// now run model
	start = nnom_ms_get();
	model_run(pre->model);
	pre->t_run_total += nnom_ms_get() - start;

	// find how many prediction is bigger than the ground true.
	// Raning rules, same as tensorflow. however, predictions in MCU is more frequencly to have equal probability since it is using fixed-point.
	// if ranking is 1, 2, =2(true), 4, 5, 6. the result will be top 3.
	// if ranking is 1, 2(true), =2, 4, 5, 6. the result will be top 2.
	// find the ranking of the prediced label.
	for (uint32_t j = 0; j < pre->label_num; j++)
	{
		if (j == label)
			continue;
		if (pre->buf_prediction[label] < pre->buf_prediction[j])
			true_ranking++;
		// while value[label] = value[j]. only when label > j, label is the second of j
		else if (pre->buf_prediction[label] == pre->buf_prediction[j] && j < label)
			true_ranking++;
	}

	if (true_ranking < pre->top_k_size)
		pre->top_k[true_ranking]++;

	// Find top 1 and return the current prediction.
	// If there are several maximum prediction, return the first one.
	max_val = pre->buf_prediction[0];
	max_index = 0;
	for (uint32_t j = 1; j < pre->label_num; j++)
	{
		if (pre->buf_prediction[j] > max_val)
		{
			max_val = pre->buf_prediction[j];
			max_index = j;
		}
	}

	// fill confusion matrix
	pre->confusion_mat[label * pre->label_num + max_index] += 1;

	// prediction count
	pre->predic_count++;

	// return the prediction
	return max_index;
}

void prediction_end(nnom_predic_t *pre)
{
	if (!pre)
		return;
	pre->t_predic_total = nnom_ms_get() - pre->t_predic_start;
}

void predicetion_delete(nnom_predic_t *pre)
{
	if (!pre)
		return;
	nnom_free(pre);
}

void prediction_matrix(nnom_predic_t *pre)
{
	if (!pre)
		return;
	// print titles
	printf("\nConfusion matrix:\n");
	printf("predic");
	for (int i = 0; i < pre->label_num; i++)
	{
		printf("%6d", i);
	}
	printf("\n");
	printf("actual\n");
	// print the matrix
	for (int i = 0; i < pre->label_num; i++)
	{
		uint32_t row_total = 0;

		printf(" %3d |", i);
		for (int j = 0; j < pre->label_num; j++)
		{
			row_total += pre->confusion_mat[i * pre->label_num + j];
			printf("%6d", pre->confusion_mat[i * pre->label_num + j]);
		}
		printf("   |%4d%%\n", pre->confusion_mat[i * pre->label_num + i] * 100 / row_total);
		row_total = 0;
	}
	printf("\n");
}

// top-k
void prediction_top_k(nnom_predic_t *pre)
{
	uint32_t top = 0;
	if (!pre)
		return;

	for (int i = 0; i < pre->top_k_size; i++)
	{
		top += pre->top_k[i];
		if (top != pre->predic_count)
			printf("Top %d Accuracy: %.2f%% \n", i + 1, ((float)top * 100) / pre->predic_count);
		else
			printf("Top %d Accuracy: 100%% \n", i + 1);
	}
}

// this function is to print sumarry
void prediction_summary(nnom_predic_t *pre)
{
	if (!pre)
		return;
	// sumamry
	printf("\nPrediction summary:\n");
	printf("Test frames: %d\n", pre->predic_count);
	printf("Test running time: %d sec\n", pre->t_predic_total / 1000);
	printf("Model running time: %d ms\n", pre->t_run_total);
	printf("Average prediction time: %d us\n", (pre->t_run_total * 1000) / pre->predic_count);
	printf("Average effeciency: %.2f ops/us\n", (double)((double)pre->model->total_ops * pre->predic_count) / ((double)pre->t_run_total * 1000));
	printf("Average frame rate: %.1f Hz\n", (float)1000 / ((float)pre->t_run_total / pre->predic_count));

	// print top-k
	prediction_top_k(pre);

	// print confusion matrix
	prediction_matrix(pre);
}

// stand alone prediction API
// this api test one set of data, return the prediction
// input the model's input and output bufer
// return the predicted label
int32_t nnom_predic_one(nnom_model_t *m, int8_t *input, int8_t *output)
{
	int32_t max_val, max_index;

	if (!m)
		return NN_ARGUMENT_ERROR;

	// copy data to input buf if the data is not in the same physical memory address
	if (input != m->head->in->mem->blk)
		memcpy(m->head->in->mem->blk, input, shape_size(&m->head->in->shape));

	model_run(m);

	if (output != m->tail->out->mem->blk)
		memcpy(output, m->tail->out->mem->blk, shape_size(&m->tail->out->shape));

	// Top 1
	max_val = output[0];
	max_index = 0;
	for (uint32_t i = 1; i < shape_size(&m->tail->out->shape); i++)
	{
		if (output[i] > max_val)
		{
			max_val = output[i];
			max_index = i;
		}
	}
	return max_index;
}

static void layer_stat(nnom_layer_t *layer)
{
	// layer stat
	printf(" %10s -    %6d        %7d      ",
		   (char *)&default_layer_names[layer->type],
		   layer->stat.time,
		   layer->stat.macc);

	if (layer->stat.macc != 0)
		printf("%d.%02d\n", layer->stat.macc / layer->stat.time, (layer->stat.macc * 100) / (layer->stat.time) % 100);
	else
		printf("\n");
}

void model_stat(nnom_model_t *m)
{
	size_t total_ops = 0;
	size_t total_time = 0;
	nnom_layer_t *layer;
	size_t run_num = 0;

	if (!m)
		return;

	layer = m->head;

	printf("\nPrint running stat..\n");
	printf("Layer(#)        -   Time(us)      ops(MACs)     ops/us \n");
	printf("--------------------------------------------------------\n");
	while (layer)
	{
		run_num++;
		printf("#%-3d", run_num);
		total_ops += layer->stat.macc;
		total_time += layer->stat.time;
		layer_stat(layer);
		if (layer->shortcut == NULL)
			break;
		layer = layer->shortcut;
	}
	printf("\nSummary.\n");
	printf("Total ops (MAC): %d\n", total_ops);
	printf("Prediction time :%dus\n", total_time);
	printf("Efficiency %d.%02d ops/us\n",
		   (total_ops / total_time),
		   (total_ops * 100) / (total_time) % 100);
}
