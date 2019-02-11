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
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>
#include "math.h"
#include "nnom.h"

const char default_layer_names[][12] = DEFUALT_LAYER_NAMES;
const char default_activation_names[][8] = ACTIVATION_NAMES;
size_t memory_taken = 0;

void *nnom_mem(size_t size)
{
	void *p = nnom_malloc(size);
	if(p)
	{
		memory_taken+=size; //test
		nnom_memset(p, 0, size);
	}
	return p;
}

size_t nnom_mem_stat(void )
{
	return memory_taken;
}

// get the size of an module model
static size_t io_mem_size(nnom_layer_io_t* io)
{
	size_t size = 0;
	if(io != NULL)
	{
		while(io)
		{
			size += shape_size(&io->shape);
			io = io->aux;
		}
	}
	return size;
}

size_t alignto(size_t value, uint32_t alignment)
{
	if(value%alignment == 0)
		return value;
	value += alignment - value % alignment;
	return value;
}

static nnom_layer_t* find_last(nnom_layer_t* layer)
{
	if(layer ==NULL) return NULL;
	// iterate every layer until the last one on the list, then return the last instance
	while(layer->out->hook.io != NULL)
		layer = layer->out->hook.io->owner;
	return layer;
}
// input start layer, return layer num
static uint32_t find_index(nnom_layer_t* start, nnom_layer_t* layer)
{
	uint32_t i = 1;
	if(start ==NULL) return 0;
	// iterate every layer until the last one on the list, then return the last instance
	while(start->out->hook.io != NULL)
	{
		i ++;
		if(layer == start)
			return i;
		start = start->out->hook.io->owner;
	}
	return 0;
}

static nnom_status_t model_add(nnom_model_t* model,  nnom_layer_t *layer)
{
	nnom_layer_t* last = NULL;
	nnom_layer_t* curr = NULL;
	
	if(layer == NULL)
	{	
		LOG("Error: added a NULL layer, could be no memory while creating layer.\n");
		return NN_NO_MEMORY;
	}
	
	last = find_last(model->head);
	curr = layer;
	
	// when the list is empty, the find_last() return list->head. 
	if(last == NULL) 
	{
		model->head = curr;
	}
	else
	{
		// hook the current layer with the last layer. 
		last->out->hook.io 	 = curr->in;		// hook IO
		curr->in->hook.io 	 = last->out;
	}
	return NN_SUCCESS;
}

// hook the current layer to the input layer
// this function only to connect (single output layer) to (single input layer). 
static nnom_layer_t * model_hook(nnom_layer_t* curr, nnom_layer_t *last)
{
	if(last == NULL || curr == NULL)
		return NULL;
	
	// check if the output hook is empty and can be use by us
	if(last->out->hook.io == NULL)
	{	
		// hook the io in both layer
		last->out->hook.io 	 = curr->in; 
		curr->in->hook.io 	 = last->out;
	}
	// if the output of last is already hooked. we need to create a new hook.
	else
	{
		nnom_layer_hook_t *hook = &last->out->hook;
		nnom_layer_hook_t *new_hook = NULL;
		while(hook->next != NULL)
		{
			hook = hook->next;
		}
		new_hook = nnom_mem(sizeof(nnom_layer_hook_t));
		if(new_hook == NULL)
			return NULL;
		hook->next = new_hook;
		
		// now connect them 
		// the new_hook is the new hook allocate in last layer. 
		new_hook->io      = curr->in; //primary IO. 
		curr->in->hook.io = last->out;
	}
	return curr;
}


// find an available hook on the io module, normally used by output io module. 
// input, the output io module that wants to hook on
// output, the new hook that added to the end of the hook list on the io 
static nnom_layer_hook_t * allocate_hook(nnom_layer_io_t * io)
{
	nnom_layer_hook_t * hook;
	if(io == NULL)
		return NULL;
	hook = &io->hook;
	
	// if the primary hook is empty, reture it directly. 
	if(hook->io == NULL)
	{
		return hook;
	}
	else
	{
		// find the empty place and allocate new hook for us
		while(hook->next!= NULL)
		{
			hook = hook->next;
		}
		hook->next = nnom_mem(sizeof(nnom_layer_hook_t));
		if(hook->next == NULL)
			return NULL;
		return hook->next;
	}
}

// to check if an input io is hooked to other layer
// input the primary io of a layer's input
// return, the new io that added to the input io list. 
static nnom_layer_io_t * allocate_input_io(nnom_layer_io_t * io)
{
	if(io == NULL)
		return NULL;
	
	// if the io is free to used
	if(io->hook.io == NULL)
	{
		return io;
	}
	else
	{
		// find the empty place and allocate new hook for us
		while(io->aux!= NULL)
		{
			io = io->aux;
		}
		io->aux = nnom_mem(sizeof(nnom_layer_io_t));
		if(io->aux == NULL)
			return NULL;
		// the owner for new io is inherited
		io->aux->owner = io->owner;
		return  io->aux;
	}
}


// merge a few layers using specified method 
// num = the number of layer that will be merged
// method = functional layer such as (concat(), dot(), mult(), add())
static nnom_layer_t * model_mergex(nnom_layer_t *method, int num, ...)
{
	nnom_layer_t * in_layer;
	nnom_layer_io_t * method_in_io;
	nnom_layer_hook_t *output_io_hook;
	
	va_list valist;
	
	if(method == NULL)
		return NULL;
	
	va_start(valist, num);
	
	for (int i = 0; i < num; i++)
    {
		// get the input layer
		in_layer = va_arg(valist, nnom_layer_t*);
		
		// add a new hook to the output io of the input layer
		output_io_hook = allocate_hook(in_layer->out);
		// add a new input io to the method layer's input list.
		method_in_io = allocate_input_io(method->in); 
		
		// manually hook them togeter. 
		output_io_hook->io = method_in_io;			
		method_in_io->hook.io = in_layer->out;		
    }
	va_end(valist);
	return method;
}

// merge 2 input 
// this is an older interface
// method = functional layer such as (concat(), dot(), mult(), add())
static nnom_layer_t * model_merge(nnom_layer_t *method, nnom_layer_t *in1, nnom_layer_t *in2)
{
	return model_mergex(method, 2, in1, in2);
}


// This api will merge activation to layer's to reduce the extra layer for activation
static nnom_layer_t * model_active(nnom_activation_t* act, nnom_layer_t * target)
{
	// simple and easy
	target->actail = act;
	return target;
}



// when model=NULL, it create a new sequential model
nnom_model_t* new_model(nnom_model_t *model)
{
	nnom_model_t* m = model;
	if(m == NULL)
	{
		m = nnom_mem(sizeof(nnom_model_t));
		m->is_alloc = true;
	}
	else
	{
		memset(m, 0, sizeof(nnom_model_t));
		m->is_alloc = false;
	}
	
	// set methods
	m->add = model_add;
	m->hook = model_hook;
	m->merge = model_merge;
	m->mergex = model_mergex;
	m->active = model_active;
	
	return m;
}

void model_delete(nnom_model_t* m)
{
	// free all mem in list first
	
	// free model, if the model is created by nnom
	if(m->is_alloc)
		nnom_free(m);
}


// find and available memory block. 
static nnom_mem_block_t * allocate_block(nnom_mem_block_t *list)
{
	nnom_mem_block_t * free = NULL;
	uint32_t idx;
	
	for(idx = 0; idx < NNOM_BLOCK_NUM; idx ++)
	{
		if(list[idx].owners == 0)
			break;
	}
	free = &list[idx];
	return free;
}

static void release_block(nnom_mem_block_t *block)
{
	if(block->owners > 0)
		block->owners -= 1;
	if(block->owners == 0)
		block->state = NNOM_BUF_EMPTY; 			// test, marked empty
}

static void release_input_mem(nnom_layer_t *layer)
{
	nnom_layer_io_t *in;
	// release all input of buf
	in = layer->in;
	while(in != NULL)
	{
		release_block(in->mem);
		in = in->aux;
	}
}
static void release_comp_mem(nnom_layer_t *layer)
{
	// release computational buf if exist
	if(layer->comp != NULL)
	{
		release_block(layer->comp->mem);
	}
}

static uint8_t hook_length(nnom_layer_hook_t * hook)
{
	uint8_t num = 0;
	if(hook == NULL) return 0;
	while(hook != NULL)
	{
		num ++;
		hook = hook->next;
	}
	return num;
}

// call while iteration. the shorcut is for fast running and fast iliterating. 
// simply link every layer as a list. ordered by its runing order
nnom_status_t layer_shortcut_add(nnom_layer_t *start, nnom_layer_t* curr)
{
	nnom_layer_t* layer = start; 
	// first one, return 
	if(start == curr)
	{
		return NN_SUCCESS;
	}
	// find the end of the list, and add curr layer to the end of it. 
	while(layer->shortcut != NULL)
	{
		// if the layer is alread in shortcut list, tell upper. 
		if(curr == layer)
			return NN_LENGTH_ERROR;
			
		layer = layer->shortcut;
	}
	layer->shortcut = curr;
	
	return NN_SUCCESS;
}

nnom_status_t compile_layers(nnom_layer_t *start, nnom_mem_block_t* block_pool)
{
	size_t mem_size = 0;
	nnom_layer_t * layer = start;
	nnom_layer_io_t *in;
	nnom_layer_io_t *out;
	nnom_layer_hook_t * hook;
	
	nnom_mem_block_t *in_blk;
	nnom_mem_block_t *out_blk;
	
	in = layer->in;
	out = layer->out;
	
	while(layer)
	{
		// check input
		in = layer->in;
		
		// check if this layer is input layer 
		// first layer has no input hooked, and is not initialized
		if(in->hook.io == NULL )
		{
			// if not initalized
			if(in->mem == NULL)
			{
				in_blk = allocate_block(block_pool);
				in_blk->owners += 1; 						// add 1
				mem_size = alignto(shape_size(&in->shape),4);
				in_blk->size = mem_size> in_blk->size ? mem_size : in_blk->size;
				// set the blk to the layer IO
				in->mem = in_blk;
				in->mem->state = NNOM_BUF_FILLED; //mark input buff filled
			}
		}
		else
		{	
			// check every in
			while(in != NULL)
			{
				in->mem = in->hook.io->mem;
				in = in->aux;
			}
		}
		
		// if there are mutiple input, wait till all block filled
		in = layer->in;
		if(in != NULL && in->aux != NULL)
		{
			while(in != NULL) 
			{
				//if(in->mem->owner != layer)
				if(in->mem == NULL || in->mem->state != NNOM_BUF_FILLED)	
					return NN_MORE_TODO;
				in = in->aux;
			}
		}

		
		// calculate output shape while all inputs are filled
		layer->comp_out_shape(layer);
		layer_shortcut_add(start, layer); 
		
		// assign for computational buf
		if(layer->comp != NULL)
		{
			layer->comp->mem = allocate_block(block_pool);
			layer->comp->mem->owners += 1;				// add users
			layer->comp->mem->state = NNOM_BUF_FILLED; 	// test
			// record maximum mem size in this block
			mem_size = alignto(shape_size(&layer->comp->shape),4);
			layer->comp->mem->size = 
				mem_size > layer->comp->mem->size ? mem_size : layer->comp->mem->size;
		}
		
		// show block size
		{
			size_t in_size = io_mem_size(layer->in);
			size_t out_size = io_mem_size(layer->out);
			size_t compsize;
			if(layer->comp != NULL)
				compsize = shape_size(&layer->comp->shape);
			else
				compsize = 0;
			LOG(" %10.s - ", &default_layer_names[layer->type]);
			if(layer->actail != NULL)
				LOG("%8.s - ", &default_activation_names[layer->actail->type]);
			else 
				LOG("         - ");
			
			LOG("(%4d,%4d,%4d)  %7d   (%5d,%5d,%5d)",
				layer->out->shape.h, layer->out->shape.w, layer->out->shape.c,
				layer->stat.macc, 
				in_size, 
				out_size,
				compsize);
	    }
		
		// show assigned blocks
		{
			LOG("   ");
			for(int i=0; i<NNOM_BLOCK_NUM; i++)
			{
				if(i % 4 == 0)
					LOG(" ");
				if(block_pool[i].owners)
					LOG("%d ", block_pool[i].owners);
				else
					LOG("- ");
			}
			LOG("\n");
		}
		
		// check output
		out = layer->out;
		if(out == NULL)
			return NN_SUCCESS;

		// if the layer is Single Output, continue the loop directly. To reduce nested level 
		if(out->aux == NULL && out->hook.next == NULL)
		{
			// single buf layer. 
			if(layer->in->type == LAYER_BUF_NULL || layer->out->type == LAYER_BUF_NULL)
			{
				// we dont release the io buf, pass to next layer directly
				layer->out->mem = layer->in->mem;
				// computational buf
				release_comp_mem(layer);
			}
			else
			{
				// not a single buf layer
				out_blk = allocate_block(block_pool);
				if(out_blk == NULL)
					return NN_NO_MEMORY;
				// set output mem owner to next layer. 
				out_blk->owners += 1; 
				out_blk->state = NNOM_BUF_FILLED; 				// marked filled
				// record maximu mem size in this block
				mem_size = alignto(shape_size(&out->shape),4);
				out_blk->size = mem_size > out_blk->size ? mem_size : out_blk->size;
				// set the blk to the layer IO
				out->mem = out_blk;
				
				// once we allocate for output, we can now release input and comput. 
				// release input mem and comp mem
				release_input_mem(layer);
				release_comp_mem(layer);	
			}
		}
		// Multiple output hooks
		else
		{
			// single buf layer will use the input buf for the first output 
			if(in->type == LAYER_BUF_NULL || out->type == LAYER_BUF_NULL)
			{
				// we dont allocate new buf, but use the input
				// the ownership will be set to next layer later 
				layer->out->mem = layer->in->mem; 
				layer->out->mem->owners = hook_length(&layer->out->hook); // set the mem lifetime. 
				// we dont release input. 
				release_comp_mem(layer);
			}
			// mutiple buf layer. (I/O uses different memory)
			else
			{
				// allocate for every output 
				out = layer->out;
				while(out != NULL && out->hook.io != NULL) // the output layer have no output IO
				{
					// assign new block
					out->mem = allocate_block(block_pool);
					if(out->mem == NULL)
						return NN_NO_MEMORY;
					// record maximum mem size in this block
					mem_size = alignto(shape_size(&out->shape), 4);
					out->mem->size = mem_size > out->mem->size ? mem_size : out->mem->size;
					// keep the block untill the last hooked layer is called.  
					out->mem->owners = hook_length(&out->hook);	// set lifetime
					out->mem->state = NNOM_BUF_FILLED; 			// test, marked filled
					
					out = out->aux;
				}
				// once we allocate for output, we can now release input and comput. 
				// release input mem and comp mem
				release_input_mem(layer);
				release_comp_mem(layer);
			}
			
			// iterate all hooked layers. 
			out = layer->out;
			while(out != NULL)
			{				
				// nested call hooked layer one by one. 
				hook = &out->hook;
				while(hook != NULL && hook->io != NULL)
				{
					nnom_status_t result;
//					// if this layer is the last one that hooked on the buf.
//					// then swith the block owner to him. 
//					if(hook->next == NULL)
//						out->mem->owners = hook->io->owner;
					
					// test, add shorcut before nested call 
					// put the "hooked layer" to the END of the list, which START at current "layer"
					result = layer_shortcut_add(layer, hook->io->owner); 
					
					// if the layer is already in the list, then it is already compiled. 
					if(result == NN_SUCCESS)
						// nested call
						compile_layers(hook->io->owner, block_pool);					
					// next hook
					hook = hook->next;
				}
				
				// next io
				out = out->aux;
				
				//
				if(out != NULL)
				{
					out_blk = allocate_block(block_pool);
					if(out_blk == NULL)
						return NN_NO_MEMORY;
			    }
			}
			
			// when all the out is called. this should stop here. 
			// once enter mutiple output iterating, the function will always return. 
			return NN_SUCCESS;
		}
		// Multiple output ended. 
		
		// return if this is output layer. 
		// the output layer's output io is hooked to nothing.  
		if(layer->out->hook.io == NULL)	
			return NN_SUCCESS;

		// single output layer, this function continue to analyse next layer.
		// switch to next layer.
		layer = layer->out->hook.io->owner;
	}
	return NN_SUCCESS;
}

size_t mem_analysis_result(nnom_model_t *m)
{
	uint32_t index;
	uint32_t total_mem = 0;
	LOG("INFO: memory analysis result\n ");
	for(index = 0; index < NNOM_BLOCK_NUM; index++)
	{
		total_mem += m->blocks[index].size;
		LOG("Block%d: %d  ", index, m->blocks[index].size);
	}
	LOG("\n Total memory cost by network: %d bytes\n", total_mem);
	
	return  total_mem;
}

nnom_status_t block_mem_set(nnom_model_t * m, void * buf)
{
	uint32_t index;
	uint32_t mem_offset = 0;
	
	for(index = 0; index < NNOM_BLOCK_NUM; index++)
	{
		if(m->blocks[index].size == 0)
			break;
		m->blocks[index].blk = (void*)((uint32_t)buf + mem_offset);
		mem_offset += m->blocks[index].size;
	}
	return NN_SUCCESS;
}

// this function has to be use after memory is assigned to the layers. 
// it means it has to be call after compile_model as well. 
// it simply get the output buffer and set the buffer to tailed activation of each layer.. 
nnom_status_t set_tailed_activation(nnom_model_t* m)
{
	NNOM_NULL_CHECK(m);
	NNOM_NULL_CHECK(m->head);	
	nnom_layer_t * layer = m->head;
	
	// if tailed activation, set it to the output. 
	while(layer)
	{
		if(layer->actail != NULL)
		{
			layer->actail->data = layer->out->mem->blk;
			layer->actail->size = shape_size(&layer->out->shape);
			layer->actail->fmt  = layer->out->qfmt;
		}
		if(layer->shortcut == NULL)
			break;
		layer = layer->shortcut;
	}

	return NN_SUCCESS;
}


// get total ops
static uint64_t model_set_ops(nnom_model_t *m)
{
	nnom_layer_t *layer; 
	uint64_t total_ops = 0;
	layer = m->head;
	while(layer)
	{
		total_ops += layer->stat.macc;
		if(layer->shortcut == NULL)
			break;
		layer = layer->shortcut;
	}
	m->total_ops = total_ops;
	return total_ops; 
}

// a compile can be use for both sequencial / functional model. 
// the output layer is optional, if output = NULL, the compile set the 
nnom_status_t model_compile(nnom_model_t *m, nnom_layer_t* input, nnom_layer_t* output)
{
	size_t buf_size;
	uint8_t* buf;
	NNOM_NULL_CHECK(m);
	NNOM_NULL_CHECK(input);	
	
	m->head = input;
	m->tail = output;
	if(output == NULL)
		m->tail = find_last(input);
	
	LOG("\nINFO: Start compile...\n");
	LOG("Layer        Activation    output shape      ops          memory            mem life-time\n");
	LOG("----------------------------------------------------------------------------------------------\n");
	
	// compile layers, started from list head
	compile_layers(m->head, m->blocks);
	
	LOG("----------------------------------------------------------------------------------------------\n");
	
	// if model's tail is not the last layer which built by user. 
	if(output != find_last(input))
		LOG("WARNING: model returned at #%d %s layer, but this layer is not the end of shortcut list \n", 
				find_index(m->head, output), &default_layer_names[output->type]);
	
	// get the total (aligned) memory requirement
	buf_size = mem_analysis_result(m);
	
	// allocate memory
	buf = nnom_mem(buf_size);
	
	// set allocated memory for layer
	block_mem_set(m, buf);
	
	// finally set the output buff to tailed activation on each layer.
	set_tailed_activation(m);
	
	// set model total ops
	model_set_ops(m);
	
	return NN_SUCCESS;
}

// a simpler api for compile models with sequencial model
// this does not require specified Input / Output layers 
nnom_status_t sequencial_compile(nnom_model_t *m)
{
	nnom_layer_t *input, *output;
	input = m->head;
	output = find_last(input);
	return  model_compile(m, input, output);
}


nnom_status_t layer_run(nnom_layer_t *layer)
{
	nnom_status_t result; 
	uint32_t start;
	NNOM_NULL_CHECK(layer);
	
	// start
	start =  nnom_us_get();
	// run main layer first
	result = layer->run(layer);
	// check tailed-activation
	if(layer->actail != NULL)
	{
		layer->actail->run(layer, layer->actail);
	}
	// end
	layer->stat.time = nnom_us_get() - start;
	return result;
}

// run the model, until the end_layer. If end_layer == NULL, run all layer. 
nnom_status_t model_run_to(nnom_model_t *m, nnom_layer_t *end_layer)
{
	uint32_t layer_num = 1;
	nnom_status_t result;
	nnom_layer_t * layer;
	NNOM_NULL_CHECK(m);
	NNOM_NULL_CHECK(m->head);	
	
	layer = m->head;
	// using shortcut run
	while(layer)
	{
		result = layer_run(layer);
		if(result != NN_SUCCESS)
		{
			LOG("Error: #%d %s layer return error code:%d\n", layer_num, &default_layer_names[layer->type], result);
			return result;
		}
		if(layer == end_layer)
		{
			return result;
		}

		//LOG("INFO:Run - %10s, time: %d \n", &default_layer_names[layer->type], layer->stat.time);
		if(layer->shortcut == NULL)
			break;
		layer = layer->shortcut;
		layer_num ++;
	}
	
	return NN_SUCCESS;
}

// run all model. 
nnom_status_t model_run(nnom_model_t *m)
{
	return model_run_to(m, NULL);
}





