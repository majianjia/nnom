
## Layer APIs

Layers APIs are listed in *nnom_layers.h*

**Notes**

- 1D/2D operations are both working with (H, W, C) format, known as "channel last". 
- When working with 1D operations, the H for all the shapes must be 1 constantly.


---

## Input()

~~~c
nnom_layer_t* Input(nnom_shape_t input_shape, * p_buf);
~~~

**A model must start with a Input layer** to copy input data from user memory space to NNoM memory space. 
If NNoM is set to CHW format, the Input layer will also change the input format from HWC (regular store format for image in memory) to CHW during copying. 

**Arguments**

- **input_shape:** the shape of input data to the model. 
- **p_buf:** the data buf in user space. 

**Return**

- The layer instance

---

## Output()

~~~c
nnom_layer_t* Output(nnom_shape_t output_shape* p_buf);
~~~

Output layer is to copy the result from NNoM memory space to user memory space. 

**Arguments**

- **output_shape:** the shape of output data. (might be deprecated later)
- **p_buf:** the data buf in user space. 

**Return**

- The layer instance.

---

## Conv2D()

~~~c
nnom_layer_t* Conv2D(uint32_t filters, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad,
	nnom_weight_t *w, nnom_bias_t *b);
~~~

This funtion is for 1D or 2D, mutiple channels convolution.  

**Arguments**

- ** filters:** the number of filters. or the channels of the output spaces.
- **k (kernel):** the kernel shape, which is returned by `kernel()`
- **s (stride):** the stride shape, which is returned by `stride()`
- **pad (padding):** the padding method either `PADDING_SAME` or `PADDING_VALID`
- **w (weights) / b (bias)**: weights and bias constants and shits. Generated in `weights.h`

**Return**

- The layer instance


**Notes**

When it is used for 1D convolution, the H should be set to 1 constantly in kernel and stride.  


---
 
## DW_Conv2D()

~~~C
nnom_layer_t* DW_Conv2D(uint32_t multiplier, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad, 
	nnom_weight_t *w, nnom_bias_t *b);
~~~

This funtion is for 1D or 2D, mutiple channels depthwise convolution.  

**Arguments**

- ** mutiplier:** the number of mutiplier. **Currently only support mutiplier=1 **
- **k (kernel):** the kernel shape, which is returned by `kernel()`
- **s (stride):** the stride shape, which is returned by `stride()`
- **pad (padding):** the padding method either `PADDING_SAME` or `PADDING_VALID`
- **w (weights) / b (bias)**: weights and bias constants and shits. Generated in `weights.h`


**Return**

- The layer instance

**Notes**

When it is used for 1D convolution, the H should be set to 1 constantly in kernel and stride.  

---

## Dense()

~~~C
nnom_layer_t* Dense(size_t output_unit, nnom_weight_t *w, nnom_bias_t *b);
~~~

A fully connected layer. It will flatten the data if the last output is mutiple-dimensions. 

**Arguments**

- ** output_unit:** the number of output unit.
- **w (weights) / b (bias)**: weights and bias constants and shits. 

**Return**

- The layer instance

---

## UpSample()

~~~C
nnom_layer_t *UpSample(nnom_shape_t kernel);
~~~

A basic up sampling, using nearest interpolation

**Arguments**

- **kernel:** a shape object returned by `kernel()`, the interpolation size.

**Return**

- The layer instance

---

## ZeroPadding()

~~~C
nnom_layer_t *ZeroPadding(nnom_border_t pad);
~~~

Pad zeros to the image for each edge (top, bottom, left, right)

**Arguments**

- **pad:** a border object returned by `border()`, contains top, bottom, left and right padding.

**Return**

- The layer instance

---

## Cropping()

~~~C
nnom_layer_t *Cropping(nnom_border_t pad);
~~~

It crops along spatial dimensions.

**Arguments**

- **pad:** a border object returned by `border()`, contains top, bottom, left and right size.

**Return**

- The layer instance

---

## Lambda()

~~~C

// layer.run()   , compulsory
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

- **`(*run)(nnom_layer_t *)`:** or so called run method, is the method to do the customized operation. 
- **`(*oshape)(nnom_layer_t *)`:** is to calculate the output shape according to the input shape during compiling. If this method is not presented, the input shape will be passed to the output shape.   
- **`(*free)(nnom_layer_t *)`:** is to free the resources allocated by the users. This method will be called when the model is deleting. Leave it NULL if no resources need to be released. 
- **parameters:** is the pointer to user configurations. User can access to it in all three methods above.

**Return**

- The layer instance

**Notes**

- All methods with type `nnom_status_t` must return `NN_SUCCESS` to allow the inference process. Any return other than that will stop the inference of the model. 
- When `oshape()` is presented, please refer to examples of other similar layers. The shape passing must be handle carefully.
- This method is called in compiling, thus it can also do works other than calculating output shape only. An exmaple is the `global_pooling_output_shape()` fills in the parameters left by `GlobalXXXPool()`

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
layer = DW_Conv2D(1, kernel(3, 3), stride(1, 1), PADDING_VALID, &conv2d_3_w, &conv2d_3_b);`
~~~

** Dense:**
~~~C
nnom_layer_t *layer;
layer = Dense(32, &dense_w, &dense_b);
~~~

** UpSample:**
~~~C
nnom_layer_t *layer;
layer = UpSample(kernel(2, 2)); // expend the output size by 2 times in both H and W axis. 
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





















