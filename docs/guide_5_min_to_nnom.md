

# 5 min Guide to NNoM

The aim for NNoM is to help **Embedded Engineers** to develop and deploy **Neural Network models** onto the **MCUs**. NNoM is working closely with **Keras**. 
If you dont know Keras yet **[Getting started: 30 seconds to Keras](https://keras.io/#getting-started-30-seconds-to-keras)** 

This guide will show you how to use NNoM for your very first step from an embedded engineer perspective. 

---

## Backgrouds Checking 

You should:

- know C language and your target MCU enviroment. 
- know a bit of python.

You must **NOT**:

- be a pro in TensorFlow / lite :-)

---

## Neural Network with Keras

If you know nothing about Keras, you must check **[Getting started: 30 seconds to Keras](https://keras.io/#getting-started-30-seconds-to-keras)** first.

Lets say if we want to classify the MNIST hand writing dataset.
This is what you normally do with Keras. 

~~~Python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
~~~

Each operation in Keras are defined by "Layer", same as we did in NNoM. The terms are different from Tensorflow (That is why you must not be a PRO in Tensorflow >_<). 

This model is with an input dimension 784, a hidden fully connected layer including 32 units and outputing 10 units(which is the number of classification(number 0~9)). 
The hidden layer is activated by ReLU activation (which keep all possitive values while set all nagtive values to 0). 

After you have trained this `model` using the method in the Keras' guide, the `model` can now do prediction. 
If you feed new image to it, it will tell you what is the wrtten number. 

Please try to run a example in Keras or NNoM if you are still confusing. 

---

## Deployed using NNoM

After the `model` is trained, the weights and parameters are already functional. We can now convert it to C language files then put it in your MCU project. 

> The result of this step is a single `weights.h` file, which contains everything you need.

To conver the model, NNoM has provided an simple API `generate_model()`[API](api_nnom_utils.md) to automaticly do the job. 
Simply pass the `model` and the test dataset to it. 
It will do all the magics for you. 

~~~Python
generate_model(model, x_test, name='weights.h')
~~~

When the conversion is finished, you will find a new `weights.h` under your working folder. 
Simply copy the file to your MCU project, and call `model = nnom_model_create();` inside you `main()`. 

Below is what you should do in practice. 

~~~C
#include "nnom.h"
#include "weights.h"

int main(void)
{
	nnom_model_t *model;
	
	model = nnom_model_create();
	model_run(model);
}
~~~

Then, your model is now running on you MCU. 
If you have supported `printf` on your MCU, you should see the compiling info on your consoles. 

Compiling logging similar to this:

~~~
Start compiling model...
Layer(#)         Activation    output shape    ops(MAC)   mem(in, out, buf)      mem blk lifetime
-------------------------------------------------------------------------------------------------
#1   Input      -          - (  28,  28,   1)          (   784,   784,     0)    1 - - -  - - - - 
#2   Conv2D     - ReLU     - (  28,  28,  12)      84k (   784,  9408,    36)    1 1 3 -  - - - - 
#3   MaxPool    -          - (  14,  14,  12)          (  9408,  2352,     0)    1 2 3 -  - - - - 
#4   UpSample   -          - (  28,  28,  12)          (  2352,  9408,     0)    1 2 2 -  - - - - 
#5   Conv2D     -          - (  14,  14,  12)     254k (  2352,  2352,   432)    1 1 2 1  1 - - - 
#6   Conv2D     -          - (  28,  28,  12)    1.01M (  9408,  9408,   432)    1 1 2 1  1 - - - 
#7   Add        -          - (  28,  28,  12)          (  9408,  9408,     0)    1 1 1 1  1 - - - 
#8   MaxPool    -          - (  14,  14,  12)          (  9408,  2352,     0)    1 1 1 2  1 - - - 
#9   Conv2D     -          - (  14,  14,  12)     254k (  2352,  2352,   432)    1 1 1 2  1 - - - 
#10  AvgPool    -          - (   7,   7,  12)          (  2352,   588,   168)    1 1 1 1  1 1 - - 
#11  AvgPool    -          - (  14,  14,  12)          (  9408,  2352,   336)    1 1 1 1  1 1 - - 
#12  Add        -          - (  14,  14,  12)          (  2352,  2352,     0)    1 1 - 1  1 1 - - 
#13  MaxPool    -          - (   7,   7,  12)          (  2352,   588,     0)    1 1 1 2  - 1 - - 
#14  UpSample   -          - (  14,  14,  12)          (   588,  2352,     0)    1 1 - 2  - 1 - - 
#15  Add        -          - (  14,  14,  12)          (  2352,  2352,     0)    1 1 1 1  - 1 - - 
#16  MaxPool    -          - (   7,   7,  12)          (  2352,   588,     0)    1 1 1 1  - 1 - - 
#17  Conv2D     -          - (   7,   7,  12)      63k (   588,   588,   432)    1 1 1 1  - 1 - - 
#18  Add        -          - (   7,   7,  12)          (   588,   588,     0)    1 1 1 -  - 1 - - 
#19  Concat     -          - (   7,   7,  24)          (  1176,  1176,     0)    1 1 1 -  - - - - 
#20  Dense      - ReLU     - (  96,   1,   1)     112k (  1176,    96,  2352)    1 1 1 -  - - - - 
#21  Dense      -          - (  10,   1,   1)      960 (    96,    10,   192)    1 1 1 -  - - - - 
#22  Softmax    -          - (  10,   1,   1)          (    10,    10,     0)    1 - 1 -  - - - - 
#23  Output     -          - (  10,   1,   1)          (    10,    10,     0)    1 - - -  - - - - 
-------------------------------------------------------------------------------------------------
Memory cost by each block:
 blk_0:9408  blk_1:9408  blk_2:9408  blk_3:9408  blk_4:2352  blk_5:588  blk_6:0  blk_7:0  
 Total memory cost by network buffers: 40572 bytes
Compling done in 76 ms

~~~

You can now use the model to predict your data. 

- Firstly, filling the input buffer `nnom_input_buffer[]` with your own data(image, signals) which is defined in `weights.h`. 
- Secondly, call `model_run(model);` to do your prediction. 
- Thirdly, read your result from `nnom_output_buffer[]`. The maximum number is the results. 

Now, please do check NNoM examples for more fancy methods. 

---

## What's More?

To be continue..

















