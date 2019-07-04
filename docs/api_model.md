
## Model APIs

NNoM support **Sequential model** and **Functional model** whitch are similar to Keras. 

Model is a minimum runable object in NNoM. 

Here list the Model APIs that used for create, compile and run a model. 

---

## new_model()

~~~C
nnom_model_t *new_model(nnom_model_t *m);
~~~

This method is to create or initiate a model instance. 

**Arguments**

- ** m:** the model instance that need to be initiated. If `NULL` is passed to it, the method will create a new model instance. 

**Return**

- The created or the initiated model instance.

---

## model_delete()

~~~C
void model_delete(nnom_model_t *m);  
~~~

Delete and free all the resources created with the model.

**Arguments**

- ** m:** the model instance.

---
## sequencial_compile()

~~~C
nnom_status_t sequencial_compile(nnom_model_t *m);
~~~

Compile a sequencial model which is constructed by sequencial construction APIs.

**Arguments**

- ** m:** the model instance for compile.

**Return**

- Status of compiling. 

---

## model_compile()

~~~C
nnom_status_t 	model_compile(nnom_model_t *m, nnom_layer_t* input, nnom_layer_t* output);
~~~

Compile a functional model which is constructed by functional construction APIs. 

**Arguments**

- ** m:** the model instance for compile.
- ** input:** the specified input layer instance.
- ** output:** the specified output layer instance. If left `NULL`, the all layers will be compile.

**Return**

- Status of compiling. 

---

## model_run()

~~~C
nnom_status_t 	model_run(nnom_model_t *m);
~~~

To run all the layers inside the model. Run one prediction. 

**Arguments**

- ** m:** the model instance for compile.
- ** input:** the specified input layer instance.
- ** output:** the specified output layer instance. If left `NULL`, the all layers will be compile.

**Return**

- The status of layer running. 

**Note**

User must fill in the input buffer which has passed to the input layer before run the model. 
The input layer then copy the data from user space to NNoM memory space to run the model. 
The results of prediction will be copy from NNoM memory space to user memory space by Output layer. 

---

## model_run_to()

~~~C
nnom_status_t model_run_to(nnom_model_t *m, nnom_layer_t *end_layer);
~~~

Same as `model_run()` but it only run partly to the specified layer. 

**Arguments**

- ** m:** the model instance for compile.
- ** end_layer:** the layer where to stop.

**Return**

- The result of layer running. 

---

## (*layer_callback)()

~~~C
nnom_status_t (*layer_callback)(nnom_model_t *m, nnom_layer_t *layer);
~~~

This callback is a runtime callback, which can then be used to evaluate the performance, extract the intermediate output etc. 
It will be called after each layer has ran. (if a actail (activation tail) is present, this callback will be called after the actail)


**Arguments**

- ** m:** the model instance of the current model.
- ** layer:** the layer instance which has just been ran.

**Return**

- The result of the callback. Any result other than NN_SUCCESS will cause the model to return immediately with the error code.  

**NOTE**

- You should not change ANY setting inside the model instance or the layer instance unless you know what you are doing. 
All configurations and buffers must be read-only inside this method. 

- This is a runtime callback which means it will affect your performance if this callback is time consuming. 

---

## model_set_callback()

~~~C
nnom_status_t model_set_callback(
	nnom_model_t *m, 
	nnom_status_t (*layer_callback)(nnom_model_t *m, nnom_layer_t *layer));
~~~

Set a callback to model. Please refer to the `(*layer_callback)()`. 
If a callback is already set but not the same one as what is given, this method will return error. 
You need to delete the old callback by `model_delete_callback()` before setting new callback.

**Arguments**

- ** m:** the model instance to be set.
- ** *layer_callback:** the layer callback.

**Return**

- NN_SUCCESS: when callback is set successfully
- NN_LENGTH_ERROR: when callback is existed in the model but not the same as the given one. 

---

## model_delete_callback()

~~~C
void model_delete_callback(nnom_model_t *m);
~~~

Delete an existing callback on the model.

**Arguments**

- ** m:** the model instance to delete the callback from.

---


## Examples

This example shows a 2 layer MPL for MNIST dateset. Input shape 28 x 28 x 1 hand writing image. 
Please check [mnist-densenet example](https://github.com/majianjia/nnom/tree/master/examples/mnist-densenet) for further reading. 

**Sequential model**

~~~C

/* nnom model */
int8_t input_data[784];
int8_t output_data[10];
void sequencial_model(void)
{
	nnom_model_t model;

	new_model(&model);
	
	model.add(&model, Input(shape(784, 1, 1), input_data));
	model.add(&model, Flatten());
	model.add(&model, Dense(100, &w1, &b1));
	model.add(&model, Dense(10, &w2, &b2));
	model.add(&model, Softmax())
	model.add(&model, Output(shape(10, 1, 1), output_data));

	sequencial_compile(&model);
	
	while(1)
	{
		feed_data(&input_data);
		model_run(&model);
		
		// evaluate on output_data[]
		...
	}
}

~~~

**Functional model**

~~~C

/* nnom model */
int8_t input_data[784];
int8_t output_data[10];
void functional_model(void)
{
	static nnom_model_t model;
	nnom_layer_t *input, *x;

	new_model(&model);
	
	input = Input(shape(784, 1, 1), input_data);
	x = model.hook(Flatten(), input);
	x = model.hook(Dense(100, &w1, &b1), x)
	x = model.hook(Dense(10, &w2, &b2), x)
	x = model.hook(Softmax(), x)
	x = model.hook(Output(shape(10, 1, 1), output_data), x);

	// compile these layers into the model. 
	model_compile(&model, input, x);

	while(1)
	{
		feed_data(&input_data);
		model_run(&model);
		
		// evaluate on output_data[]
		...
	}
}

~~~

**Layer Callback**

~~~C

// this callback to print output size of every layer.
nnom_status_t callback(nnom_model_t *m, nnom_layer_t *layer)
{
	printf("layer %s, output size %d \n", default_layer_names[layer->type], shape_size(&layer->out->shape));
	return NN_SUCCESS;
}

int main(void)
{
	// using automatic tools to generate model
	model = nnom_model_create();
	
	// set callback
	model_set_callback(model, callback);
	
	// run and see what happend.
	model_run(model);
}

~~~

Here is the console logging of the callback output:
~~~
layer Input, output shape 784 
layer Conv2D, output shape 9408 
layer MaxPool, output shape 2352 
layer UpSample, output shape 9408 
layer Conv2D, output shape 2352 
layer Conv2D, output shape 9408 
layer Add, output shape 9408 
layer MaxPool, output shape 2352 
layer Conv2D, output shape 2352 
...

~~~









































 
 


















































