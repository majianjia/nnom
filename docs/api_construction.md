
## Constructions APIs

NNoM support both **Sequential model** and **Functional model** similar to Keras. 
NNoM treat them equaly in compiling and running, the only difference the methods of constructions. 

In **Sequential model**, the layer are stacked one by one sequently using `model.add()`. So call Sequential API. 

In **Functional model**, the links between layer are specified explicitly by using `model.hook()`, `model.merge()` or `model.mergex()`. So call Functional APIs

---

# Sequential API

---

## model.add()

~~~c
nnom_status_t model.add(nnom_model_t *model,  nnom_layer_t *layer);
~~~

It is the only sequencial constructor. It stacks the new layer to the rear of the existing model. 

**Arguments**

- ** model:** the model to stack new layer.
- ** layer:** the new layer instance to stack onto the model.

**Return**

- The state of the operation. 

**Example to stack two layers on a model**
~~~c
	model.add(&model, Conv2D(16, kernel(1, 9), stride(1, 2), PADDING_SAME, &c1_w, &c1_b));
	model.add(&model, ReLU());
	...
~~~

**Notes**

- The first layer for a model must be **Input layer**. The last layer could be the **Output layer**
- You can stack like this whenever there are memory available :-)


---

# Functional APIs


---

## model.hook()
 
~~~C
nnom_layer_t *model.hook(nnom_layer_t *curr, nnom_layer_t *last);
~~~

A functional constructor to explicitly hook two layers togethers. When two layers are hook, the previous layer's output (last) will be the input of the new later (curr). 

**Arguments**

- ** curr:** the new layer for hooking to a previous built layer.
- ** last:** the previous built layer instance. 

**Return**

- The curr layer instance. 
 

**Note**

A layer instance can be hooked many times (act as "last" layer). NNoM will manage the topology and run order from them during compiling. This is very useful when many layers wants to take the same output of previous layer. The example is many towers layer share one output in Inception structure. 
 
---

## model.merge()

~~~c
nnom_layer_t *model.merge(nnom_layer_t *method, nnom_layer_t *in1, nnom_layer_t *in2);
~~~

A functional constructor for merge many layers' output by the specified merging methods(layer).

Specificaly, this method merge 2 layer's output. 

**Arguments**

- ** method:** the merging layer method. One of `Concat(), Mult(), Add(), Sub()`
- ** in1:** the first layer instance to merge. 
- ** in2:** the second layer instance to merge. 

**Return**

- The method (layer) instance. 
 
--- 
## model.mergex()

~~~c
nnom_layer_t *model.mergex(nnom_layer_t *method, int num, ...)
~~~

A functional constructor for merge many layers' output by the specified merging methods(layer).


**Arguments**

- ** method:** the merging layer method. One of `Concat(), Mult(), Add(), Sub()`
- ** num:** number of layer that needs to be merged.
- ** ...:** the list of merge layer instances. 

**Return**

- The method (layer) instance. 

**Note**
Currently, all "merge methods" support mutiple input layers, they will be processed one by one with the order provided by the list. 
 
---

## model.active()

~~~C
nnom_layer_t *model.active(nnom_activation_t *act, nnom_layer_t *target)
~~~

A functional constructor, it merges the activation to the targeted layer to avoid an redundant activation layer, which costs more memory.

**Arguments**

- ** act:** the activation instance, please check Activation for more detail.
- ** target:** the layer which output will be activated by the provided activation. 

**Return**

- The targeted (layer) instance. 

---

## Examples

This example shows the construction of an Inception model. 

~~~C
	nnom_layer_t* input_layer, *x, *x1, *x2, *x3;
	
	input_layer = Input(shape(INPUT_HIGHT, INPUT_WIDTH, INPUT_CH), nnom_input_data);

	// conv2d - 1 - inception
	x1 = model.hook(Conv2D(16, kernel(1, 5), stride(1, 1), PADDING_SAME, &c2_w, &c2_b), x);
	x1 = model.active(act_relu(), x1);
	x1 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x1);
	
	// conv2d - 2 - inception
	x2 = model.hook(Conv2D(16, kernel(1, 3), stride(1, 1), PADDING_SAME, &c3_w, &c3_b), x);
	x2 = model.active(act_relu(), x2);
	x2 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x2);
	
	// maxpool - 3 - inception
	x3 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x);
	
	// concatenate 
	x = model.mergex(Concat(-1), 3, x1, x2, x3);
	
	// flatten
	x = model.hook(Flatten(), x);
	...
~~~
 
 
 
 
 


















































