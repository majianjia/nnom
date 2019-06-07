
## Properties

Properties include some basic properties such as shape of the data buffer, Q-format of the data.

---

## Typedef

~~~C
#define nnom_shape_data_t uint16_t

typedef struct _nnom_shape
{
	nnom_shape_data_t h, w, c;
} nnom_shape_t;

typedef struct _nnom_weights
{
	const void *p_value;
	size_t shift;	// the right shift for output
} nnom_weight_t;

typedef struct _nnom_bias
{
	const void *p_value;
	size_t shift;  // the left shift for bias
} nnom_bias_t;

typedef struct _nnom_qformat
{
	int8_t n, m;
} nnom_qformat_t;

typedef struct _nnom_border_t
{
	nnom_shape_data_t top, bottom, left, right;
} nnom_border_t;

~~~

---

# Methods

---

## shape() 

~~~C
nnom_shape_t shape(size_t h, size_t w, size_t c);
~~~

**Arguments**

- ** h:** size of H, or number of row, or y axis in image. 
- ** w:** size of W, or number of column, or x axis in image.
- ** c:** size of channel. 

**Return**

- A shape instance. 

---

## kernel()

~~~C
nnom_shape_t kernel(size_t h, size_t w);
~~~

Use in pooling or convolutional layer to specified the kernel size. 

**Arguments**

- ** h:** size of kernel in H, or number of row, or y axis in image. 
- ** w:** size of kernel in W, or number of column, or x axis in image.

**Return**

- A shape instance. 

---

## stride() 

~~~C
nnom_shape_t stride(size_t h, size_t w);
~~~

Use in pooling or convolutional layer to specified the stride size. 

**Arguments**

- ** h:** size of stride in H, or number of row, or y axis in image. 
- ** w:** size of stride in  W, or number of column, or x axis in image.

**Return**

- A shape instance. 

---

## border() 

~~~C
nnom_border_t border(size_t top, size_t bottom, size_t left, size_t right);
~~~

It pack the 4 padding/cropping value to a border object. 

**Arguments**

- ** top:** the padding/cropping at the top edge of the image.
- ** bottom:** the padding/cropping at the bottom edge of the image.
- ** left:** the padding/cropping at the left edge of the image.
- ** right:** the padding/cropping at the right edge of the image.

**Return**

- A shape instance. 

---

## qformat()

~~~
nnom_qformat_t qformat(int8_t m, int8_t n);
~~~

**Arguments**

- ** m:** the integer bitwidth. 
- ** n:** the fractional bitwidth.

**Return**

- A nnom_qformat_t inistance. 


**Notes**

The Q-format within model is currently handled by Python script `nnom_utils.py`. This function will be deprecated. 

---

## shape_size()

~~~C
size_t shape_size(nnom_shape_t *s);
~~~

Calculate the size from a shape. 

size = s.h * s.w * s.c;


**Arguments**

- ** s:** the shape to calculate. 

**Return**

- The total size of the shape. 






























 
 


















































