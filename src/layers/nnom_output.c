
/*
 * Copyright (c) 2018-2019
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-07-23     Jianjia Ma   The first version
 */

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_local.h"
#include "nnom_layers.h"

nnom_status_t output_build(nnom_layer_t *layer);
nnom_status_t output_run(nnom_layer_t *layer);

nnom_layer_t *Output(nnom_shape_t output_shape, void *p_buf)
{
	// they are acturally the same.. expect the type defined
	nnom_layer_t *layer = Input(output_shape, p_buf);
	if (layer != NULL)
	{
		layer->type = NNOM_OUTPUT;
		layer->run = output_run;
		layer->build = default_build;
	}
	return layer;
}


nnom_status_t output_run(nnom_layer_t *layer)
{
	nnom_io_layer_t *cl = (nnom_io_layer_t *)layer;
	memcpy(cl->buf, layer->in->tensor->p_data, tensor_size(layer->out->tensor)); // in->memory -> user memory
	return NN_SUCCESS;
}
