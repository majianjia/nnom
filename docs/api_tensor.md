
# Tensor

In the new NNoM (ver > 0.4.0), the low-level operations are based on Tensors.

Such as activations (data), weights, bias are all tensors. 

Tensor can be const (weights/bias) placed in ROM, or dynamic(activations) allocated in RAM. 

Activations tensors between layers will still sharing their data memories like the previous memory blocks management. 
It still benefit from the memory management. 

User normally won't need to access the tensors beside the tensors of the input and output layers. 

---

## new_tensor() 

~~~C
nnom_tensor_t* new_tensor(nnom_qtype_t type, uint32_t num_dim, uint32_t num_channel);
~~~

Create a tensor instance.

**Arguments**

- **type:** tensor data quantisation type, select between `NNOM_QTYPE_PER_AXIS` and `NNOM_QTYPE_PER_TENSOR`.
- **num_dim:** number of tensor dimension
- **num_channel:** when `NNOM_QTYPE_PER_AXIS` passed as type, this is the number of the selected axis(channels). 

**Return**

- The tensor instance.

**Notes**

This does not allocated the data memory. 

---

## delete_tensor() 

~~~C
void delete_tensor(nnom_tensor_t* t);
~~~

Create a tensor instance.

**Arguments**

- **t:** tensor willing to delete


**Notes**

This does not free the data memory. 

---

## tensor_set_attr() 

~~~C
nnom_tensor_t* tensor_set_attr(nnom_tensor_t* t, 
		nnom_qformat_param_t*dec_bit, nnom_qformat_param_t *offset, nnom_shape_data_t* dim, uint32_t num_dim, uint8_t bitwidth)
~~~

set the attributes of the tensors.

**Arguments**

- **t:** tensor
- **dec_bit:** the num of bit for the fractional part in q format. An array for each channel if the tensor is per-axis quantised.
- **offset:** the zero-point offset for each channels. 
- **dim:** the dimension array of the tensor.
- **num_dim:** number of dimemsions
- **bitwidth:** bitwidth of the tensor data. -- only support 8 currently. 

**Return**

- The tensor instance.

---


## tensor_set_attr_v() 

~~~C
nnom_tensor_t* tensor_set_attr_v(nnom_tensor_t* t, 
		nnom_qformat_param_t dec_bit, nnom_qformat_param_t offset, nnom_shape_data_t* dim, uint32_t num_dim, uint8_t bitwidth);
~~~

set the attributes of the tensors by value. This interface only support per-tensor quantisation tensor. 

**Arguments**

- **t:** tensor
- **dec_bit:** the num of bit for the fractional part in q format. 
- **offset:** the zero-point offset for each channels. 
- **dim:** the dimension array of the tensor.
- **num_dim:** number of dimemsions
- **bitwidth:** bitwidth of the tensor data. -- only support 8 currently. 

**Return**

- The tensor instance.

---

## tensor_get_num_channel()

~~~C
size_t tensor_get_num_channel(nnom_tensor_t* t);
~~~

Return the data size of the tensor.

**Arguments**

- **t:** tensor

**Return**

- The data size of the tensor.

---

## tensor_cpy_attr()

~~~C
nnom_tensor_t* tensor_cpy_attr(nnom_tensor_t* des, nnom_tensor_t* src);
~~~

Copy tensor attributes.

**Arguments**

- **des:** destination tensor
- **src:** source tensor

**Return**

- The destination tensor

---

## tensor_hwc2chw_q7()

~~~C
void tensor_hwc2chw_q7(nnom_tensor_t* des, nnom_tensor_t* src);
~~~

Reorder and copy tensor data.

**Arguments**

- **des:** destination tensor
- **src:** source tensor

---

## tensor_chw2hwc_q7()

~~~C
void tensor_chw2hwc_q7(nnom_tensor_t* des, nnom_tensor_t* src);
~~~

Reorder and copy tensor data.

**Arguments**

- **des:** destination tensor
- **src:** source tensor

---



