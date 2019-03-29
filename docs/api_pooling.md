
## Pooling Layers

Pooling Layers are listed in *nnom_layers.h*

1D/2D operations are both working with (H, W, C) format, known as "channel last". When working with 1D operations, the H for all the shapes must be `1` constantly.

---

## MaxPool()

~~~c
nnom_layer_t* MaxPool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad);
~~~

This funtion is for 1D or 2D, mutiple channels max pooling.  

**Arguments**

- **k (kernel):** the kernel shape, which is returned by `kernel()`
- **s (stride):** the stride shape, which is returned by `stride()`
- **pad (padding):** the padding method `PADDING_SAME` or `PADDING_VALID`

**Return**

- The layer instance 

---

## AvgPool()

~~~c
nnom_layer_t* AvgPool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad);
~~~

This funtion is for 1D or 2D, mutiple channels average pooling.  

**Arguments**

- **k (kernel):** the kernel shape, which is returned by `kernel()`
- **s (stride):** the stride shape, which is returned by `stride()`
- **pad (padding):** the padding method `PADDING_SAME` or `PADDING_VALID`

**Return**

- The layer instance 

**Notes**

Average pooling is not recommended to use with fixed-point model (such as here). Small values will be vanished when the sum is devided. (CMSIS-NN currently does not support changing the output shifting in average pooling from input to output.) 

However, if the average pooling is the second last layer right before softmax layer, you can still use average pooling for training and then replaced by sumpooling in MCU. the only different is sumpooling (in NNoM) will not divide the sum directly but looking for a best shift (dynamic shifting) to cover the largest number. 

---

## SumPool()

~~~c
nnom_layer_t* SumPool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad);
~~~

This funtion is for 1D or 2D, mutiple channels Sum pooling. This is a better alternative to average pooling **WHEN** deploy the trained model into NNoM. The output shift for the sumpool in NNoM is dynamic, means that this pooling can only place before softmax layer.

**Arguments**

- **k (kernel):** the kernel shape, which is returned by `kernel()`
- **s (stride):** the stride shape, which is returned by `stride()`
- **pad (padding):** the padding method `PADDING_SAME` or `PADDING_VALID`

**Return**

- The layer instance 

---

## GlobalMaxPool()

~~~C
nnom_layer_t *GlobalMaxPool(void);
~~~

Global Max Pooling

**Return**

- The layer instance 

---

## GlobalAvgPool()
~~~C
nnom_layer_t *GlobalAvgPool(void);
~~~

Global Average Pooling. 

Due to the same reason as discussed in Average Pooling, it is recommended to replace this layer by `GlobalSumPool()` when the layer it she second last layer, and before the softmax layer. 

If you used `generate_model()` to deploy your keras model to NNoM, this layer will be automaticly replaced by `GlobalSumPool()` when above conditions has met. 

**Return**

- The layer instance 

---

## GlobalSumPool()

~~~C
nnom_layer_t *GlobalSumPool(void);
~~~

Global sum pooling.

**Return**

- The layer instance 

---

















