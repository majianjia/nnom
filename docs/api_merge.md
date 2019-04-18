
## Merging Methods

Merge methods (layers) are use to merge 2 or more layer's output using the methods list below. 

These methods are also layers which return a layer instance. However, they normally take two or more layer's output and "merge" them into one output. These layer instance must be passed to either `model.merge(method, in1, in2)`or `model.mergex(method, num_of_input, in1, in2, 1n3 ...)`. An example will be to concat the Inception structure. 

---

## Concat() 

~~~C
nnom_layer_t* Concat(int8_t axis);
~~~

Concatenate mutiple input on the selected axis. 

**Arguments**

- ** axis:** the axis number to concatenate in HWC format. The axis could be nagative, such as '-1' indicate the last one axis which is 'Channel'.

**Return**

- The concat layer instance

**Notes**

The concatenated axis can be different in all input layers passed to this method. Other axes must be same. 

---

## Mult() 
~~~C
nnom_layer_t* Mult(int32_t oshift);
~~~

Element wise mutiplication in all the inputs. 

This layer cannot use to merge more than 2 layer, which might cause overflowing problem. 2 `Mult()` must be used separately if willing to multiply 3 layer's output. The output shift individually of the 2 steps must be identify individually. Please check the example below for more than 2 input.

**Arguments**

- **oshift:** the output shift of this layer.  

**Return**

- The mult layer instance

**Notes**

All input layers passed to this method must have same output shape. 

---

## Add() 

~~~C
nnom_layer_t* Add(int32_t oshift);
~~~

Element wise addition in all the inputs. 

This layer cannot use to merge more than 2 layer, which might cause overflowing problem. Please refer to `Mult()`

**Arguments**

- **oshift:** the output shift of this layer.  

**Return**

- The add layer instance

**Notes**

All input layers passed to this method must have same output shape. 

---

## Sub() 

~~~C
nnom_layer_t* Sub(int32_t oshift);
~~~

Element wise substraction in all the inputs. 

This layer cannot use to merge more than 2 layer, which might cause overflowing problem. Please refer to `Mult()`

**Arguments**

- **oshift:** the output shift of this layer.  

**Return**

- The sub layer instance

**Notes**

All input layers passed to this method must have same output shape. 

---

## Example

** Channelwise concat for Inception **

~~~C
	
	input_layer = Input(shape(INPUT_HIGHT, INPUT_WIDTH, INPUT_CH), nnom_input_data);

	// conv2d - 1 - inception
	x1 = model.hook(Conv2D(16, kernel(1, 5), stride(1, 1), PADDING_SAME, &c2_w, &c2_b), x);
	x1 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x1);
	
	// conv2d - 2 - inception
	x2 = model.hook(Conv2D(16, kernel(1, 3), stride(1, 1), PADDING_SAME, &c3_w, &c3_b), x);
	x2 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x2);
	
	// maxpool - 3 - inception
	x3 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x);
	
	// concatenate 
	x = model.mergex(Concat(-1), 3, x1, x2, x3);
	
	// flatten
	x = model.hook(Flatten(), x);
	...
~~~


** Mult for 3 input ** (or Add, Sub)

In Keras

~~~python
	#instead of 
	x = multiply([x1,x2,x3]) 
	
	# you must use this instead to allow sript to calculate oshift individually. 
	x = multiply([x1,x2])
	x = multiply([x,x3])
~~~

Then in NNoM

~~~C
	// x = x1 * x2 * x3
	x = model.merge(Mult(oshift_1), x1, x2); 
	x = model.merge(Mult(oshift_2), x, x3);
~~~

