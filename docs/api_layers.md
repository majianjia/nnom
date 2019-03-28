
## Layer APIs

Layers APIs are listed in *nnom_layers.h*

1D/2D operations are both working with (H, W, C) format, known as "channel last". When working with 1D operations, the H for all the shapes must be 1 constantly.


---

## Input

~~~c
nnom_layer_t* Input(nnom_shape_t input_shape, * p_buf);
~~~

A model must start with a Input layer for copying inputd data from user space to NNoM space. 

**Arguments**

- **input_shape:** the shape of input. 
- **p_buf:** the data buf in user space. 

**Return**

- The layer instance

---

## Output

~~~c
nnom_layer_t* Output(nnom_shape_t output_shape* p_buf);
~~~

Output layer is to copy the result from NNoM space to user space. 

**Arguments**

- **output_shape:** the shape of output. 
- **p_buf:** the data buf in user space. 

**Return**

- The layer instance

---

## Conv2D

~~~c
nnom_layer_t* Conv2D(uint32_t filters, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad,
	nnom_weight_t *w, nnom_bias_t *b);
~~~

This funtion is for 1D or 2D, mutiple channels convolution.  

**Arguments**

- ** filters:** the number of filters. or the channels of the output spaces.
- **k (kernel):** the kernel shape, which is returned by `kernel()`
- **s (stride):** the stride shape, which is returned by `stride()`
- **pad (padding):** the padding method `PADDING_SAME` or `PADDING_VALID`
- **w (weights) / b (bias)**: weights and bias constants and shits. Generated in `weights.h`

**Return**

- The layer instance


**Notes**

When it is used for 1D convolution, the H should be set to 1 constantly in kernel and stride.  


---
 
## DW_Conv2D

~~~C
nnom_layer_t* DW_Conv2D(uint32_t multiplier, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad, 
	nnom_weight_t *w, nnom_bias_t *b);
~~~

This funtion is for 1D or 2D, mutiple channels depthwise convolution.  

**Arguments**

- ** mutiplier:** the number of mutiplier. **Currently only support mutiplier=1 **
- **k (kernel):** the kernel shape, which is returned by `kernel()`
- **s (stride):** the stride shape, which is returned by `stride()`
- **pad (padding):** the padding method `PADDING_SAME` or `PADDING_VALID`
- **w (weights) / b (bias)**: weights and bias constants and shits. Generated in `weights.h`


**Return**

- The layer instance

**Notes**

When it is used for 1D convolution, the H should be set to 1 constantly in kernel and stride.  


---

## Dense

~~~C
nnom_layer_t* Dense(size_t output_unit, nnom_weight_t *w, nnom_bias_t *b);
~~~

**Arguments**

- ** output_unit:** the number of output unit.
- **w (weights) / b (bias)**: weights and bias constants and shits. Generated in `weights.h`

**Return**

- The layer instance


---

## Lambda

~~~C
// Lambda Layers
// layer.run()   , required
// layer.oshape(), optional, call default_output_shape() if left NULL
// layer.free()  , optional, called while model is deleting, to free private resources
// parameters    , private parameters for run method, left NULL if not needed.
nnom_layer_t *Lambda(nnom_status_t (*run)(nnom_layer_t *),	
		nnom_status_t (*oshape)(nnom_layer_t *), 
		nnom_status_t (*free)(nnom_layer_t *),   
		void *parameters);						  
~~~

Lambda layer is an anonymous layer (interface), which allows user to do customized operation between the layer's input data and output data. 

**Arguments**
- **(*run)(nnom_layer_t *):** or so called run method, is the method to do the customized operation. 
- **(*oshape)(nnom_layer_t *):** is to calculate the output shape according to the input shape during compiling. If this method is not presented, the input shape will be passed to the output shape.   
- **(*free)(nnom_layer_t *):** is to free the resources allocated by users. this will be called when deleting models. Leave it NULL if no resources need to be released. 
- **parameters:** is the pointer to user configurations. User can access to it in all three methods above.

**Return**

- The layer instance

**Notes**

When `oshape()` is present, please refer to example of other similar layers. This method is called in compiling, thus it can also do works other than calculating output shape only. An exmaple is the `global_pooling_output_shape()` fill in the parameters left by 'GlobalXXXPool()'

---

## Examples

** Conv2D:** 
~~~C
//For 1D convolution
nnom_layer_t *layer;
layer = Conv2D(32, kernel(1, 5), stride(1, 2), PADDING_VALID, &conv2d_3_w, &conv2d_3_b);`
~~~

** DW_Conv2D:**
~~~C
nnom_layer_t *layer;
layer = Conv2D(32, kernel(3, 3), stride(1, 1), PADDING_VALID, &conv2d_3_w, &conv2d_3_b);`
~~~

** Dense:**
~~~C
nnom_layer_t *layer;
layer = Dense(32, &dense_w, &dense_b);
~~~

** Lambda:**

This example shows how to use Lambda layer to copy data from the input buffer to the output buffer. 

~~~C
nnom_status_t lambda_run(layer)
{
    memcpy(layer->output, layer->input, sizeof(inputshape);
	return NN_SUCCESS;
}

main()
{
	layer *x, *input;
	x = model.hook(Lambda(lambda_run, NULL, NULL, NULL), input); 
}
~~~





















