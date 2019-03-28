
## Model APIs

NNoM support **Sequential model** and **Functional model** whitch are similar to Keras. 

Model is a minimum runable object in NNoM. 

Here list the Model APIs that used for create, compile and run a model. 

---

## new_model

~~~C
nnom_model_t *new_model(nnom_model_t *m);
~~~

This method is to create or initiate a model instance. 

**Arguments**

- ** m:** the model instance that need to be initiated. If `NULL` is passed to it, the method will create a new model instance. 

**Return**

The created model instance or the 

---

## model_delete

~~~C
void model_delete(nnom_model_t *m);  
~~~

Delete and free all the resources created with the model.

**Arguments**

- ** m:** the model instance.

---
## sequencial_compile

~~~C
nnom_status_t sequencial_compile(nnom_model_t *m);
~~~

Compile a sequencial model which is constructed by sequencial construction APIs. 

**Arguments**

- ** m:** the model instance for compile.

**Return**

- Status of compiling. 

---

## model_compile

~~~C
nnom_status_t 	model_compile(nnom_model_t *m, nnom_layer_t* input, nnom_layer_t* output);
~~~

**Arguments**

- ** m:** the model instance for compile.
- ** input:** the specified input layer instance.
- ** output:** the specified output layer instance. If left `NULL`, the all layers will be compile.

**Return**

- Status of compiling. 

---

## model_run

~~~C
nnom_status_t 	model_run(nnom_model_t *m);
~~~

To run all the layers inside the model. 

**Arguments**

- ** m:** the model instance for compile.
- ** input:** the specified input layer instance.
- ** output:** the specified output layer instance. If left `NULL`, the all layers will be compile.

**Return**

- The result of layer running. 

**Note**

User must fill in the input buffer which has passed to the input layer before run the model. 
The input layer then copy the data from user space to NNoM memory space to run the model. 

---

## model_run_to

~~~C
nnom_status_t model_run_to(nnom_model_t *m, nnom_layer_t *end_layer);
~~~

Run partly to the specified layer. 

**Arguments**

- ** m:** the model instance for compile.
- ** end_layer:** the layer instance to stop.

**Return**

- The result of layer running. 

---
## Examples


Sequential model

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

Functional model

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










































 
 


















































