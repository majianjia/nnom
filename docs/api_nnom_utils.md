
## NNoM Utils

NNoM Utils are Python scripts for deploying models. 
What makes NNoM easy to use is the models can be deployed to MCU automatically or manually with the help of NNoM utils. 
These functions are located in `scripts/nnom_utils.py`

Tutorial is comming, before it arrives, please refer to [examples](https://github.com/majianjia/nnom/tree/dev/examples) for usage.


# API 

---

## generate_model()

~~~python
generate_model(model, x_test, name='weights.h', format='hwc', kld=True)
~~~

**This is all you need**

This method is the most frequently used function for deployment. 

1. It firsly scans the output range of each layer's output using `layers_output_ranges()`
2. Then it quantised and write the weights & bias, fused the BatchNorm parameters using `generate_weights()`
3. Finally, it generate the C functions ` nnom_model_t* nnom_model_create(void)` in `weights.h`

**Arguments**

- **model:** the trained Keras model
- **x_test:** the dataset used to check calibrate the output data quantisation range of each layer.  
- **name:** the name of the automatically generated c file. 
- **format:** indicate the backend format, options between `'hwc'` and `'chw'`. See notes
- **kld:** `True`, use KLD method for activation quantisation (saturated). `False`, use min-max method (nonsaturated). 

**Notes**

- This method might not be updated from time to time with new features in NNoM. 
- Currently, only support single input, single output models. 
- The default backend format is set to 'hwc', also call 'channel last', which is the same format as CMSIS-NN. This format is optimal for CPU. 
'chw' format, call 'channel first', is for MCU with hardware AI accelerator (such as [Kendryte K210](https://kendryte.com/)).
This setting only affects the format in the backend. the frontend will always use 'HWC' for data shape. 
- About activation quantisat method options, check [TensorRT notes](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf) for detail. 

---

## layers_output_ranges()

~~~python
layers_output_ranges(model, x_test, kld=True, calibrate_size=1000)
~~~

This function is to check the output range and generate the output shifting list of each layer. It will automatically distinguish whether a layer can change its output Q format or not. 

**Arguments**

- **model:** the trained Keras model
- **x_test:** the dataset for calibrating quantisation.
- **kld:** `True`, use KLD method for activation quantisation (saturated). `False`, use min-max method (nonsaturated). 
- **calibrate_size:** how many data for calibration. TensorRT suggest 1000 is enough. If `x_test` is longger than this value, it will randomly pick the lenght from the `x_test`.

**Return**

- The shifting list. 

**Notes**

- Checking output range of each layer is essential in deploying. It is a part of the quantisation process. 

---

## generate_weights()

~~~python
generate_weights(model, name='weights.h', format='hwc', shift_list=None)
~~~

Scans all the layer which includes weights, quantise the weights and put them into the c header.

**Arguments**

- **model:** the trained Keras model
- **name:** the c file name to store weigths.
- **shift_list:** the shift list returned by `layers_output_ranges(model, x_test)`
- **format:** indicate the backend format, options between `'hwc'` and `'chw'`. See notes in [generate_model()](#generate_model)

**Notes**

- Use function individually when willing to use none-supported operation by `generate_model()`

---


## evaluate_model()

~~~python
def evaluate_model(model, x_test, y_test, running_time=False, to_file='evaluation.txt'):
~~~

Evaluate the model after training. It do running time check, Top-k(k=1,2) accuracy, and confusion matrix. 

**Arguments**

- **model:** the trained Keras model
- **x_test:** the dataset for testing (one hot format)
- **y_test:** the label for testing dataset
- **running_time:** check running time for one prediction
- **to_file:** save above metrics to the file . 

---

## generate_test_bin()

~~~python
generate_test_bin(x, y, name='test_data_with_label.bin')
~~~

This is to generate a binary file for MCU side for model validation. 
The format of the file is shown below. 
Each batch size 128 started with 128 label, each label has converted from one-hot to number. 
The 'n' is the size of one data, such as 28x28=784 for mnist, or 32x32x3=3072 for cifar. 

|Label(0~127)|Data0|Data1|...|Data127|Label(128)|Data128...|
|---|||||||
|128-byte|n-byte|n-byte|...|n-byte|128-byte|n-bytes...|

**Arguments**

- **x:** the quantised dataset for testing
- **y:** the label for testing dataset(one hot format)
- **name:** the label for testing dataset

**Output**
- the binary file for testing on MCU.

**Notes**

The data must be quantised to the fixed-point range. For example, 
MNIST range `0~255` whcih can be converted into `0~1` using `((float)mnist/255)` for training. 
After training, it should be converted back to `0~127` for binary file because MCU only recognised q7 format. 


---

## fake_clip()

~~~python
fake_clip(frac_bit=0, bit=8)
~~~

(deprecated) This function is used for inseart a fake_quantitation using tensorflow's method `tf.fake_quant_with_min_max_vars()`, for similating quantised output of fixed-point model (the deployed model) during training. It should be inserted to the model after every 'conv' or 'dense' layer. 

However, without simulating the quantisation, most of the models are still performing well. Manually insert these layers will take huge effort for training-configuring cycle result in only a slitly better accuracy. 

**Arguments**

- **frac_bit:** the fraction bit in Q-format
- **bit:** the quantisation bitwidth. 

**Examples**

~~~python
    x = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x)
	x = fake_clip(frac_bit=7, bit=8)(x)  # quantise range to [-1~1) with 256 level. 
	x = ReLU()(x)
~~~

---

## fake_clip_min_max()

~~~python
fake_clip_min_max(min=0, max=1,  bit=8)
~~~

(deprecated) a max-min version of `fake_clip()`, check `fake_clip()` for details.

**Arguments**

- **min:** the min value to be clipped.
- **max:** the max value to be clipped. 
- **bit:** the quantisation bitwidth. 

---

# Examples

This code snips shows training using above functions. 

Please check [examples](https://github.com/majianjia/nnom/tree/master/examples) for real-life utilisation. 

~~~python
	# load data
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	
	# convert class vectors to one-hot
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
	
	# quantize the range to 0~1
    x_test = x_test.astype('float32')/255
    x_train = x_train.astype('float32')/255
	
	# (NNoM utils) generate the binary file for testing on MCU. 
	generate_test_bin(x_test*127, y_test, name='test_data.bin')
	
	# Train
	train(x_train,y_train, x_test, y_test, batch_size=128, epochs=epochs)
	
	# (NNoM utils) evaluate
    evaluate_model(model, x_test, y_test)

    # (NNoM utils) Automatically deploying, use 100 pices for output range
    generate_model(model, x_test[:100], name=weights)
	
~~~











