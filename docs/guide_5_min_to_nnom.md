

# 5 min Guide to NNoM

The aims for NNoM is to help **Embedded Engineers** to develop and deploy **Neural Network models** onto the **MCUs**. NNoM is working closely with **Keras**. 
If you dont know Keras yet **[Getting started: 30 seconds to Keras](https://keras.io/#getting-started-30-seconds-to-keras)** 

This guide will show you how to use NNoM for your very first step from an embedded engineer perspective. 

---

## Backgrouds Checking 

You should:

- know C language and your target MCU enviroment. 
- know a bit of python.

You must **not**:

- be a pro in TensorFlow / lite :-)

---

## Neural Network with Keras

If you know noting about Keras, you must check **[Getting started: 30 seconds to Keras](https://keras.io/#getting-started-30-seconds-to-keras)** 


Lets say if we want to classify the MNIST hand writing dataset.
This is what you do with Keras. 

~~~Python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
~~~

This is the model which include an input dimension 784, a hidden fully connected layer including 32 units, it is then activated by ReLU activation (which keep all possitive values while set all nagtive values to 0). Then following an output layer with 10 units, which is the number of classification(number 0~9).

After you train this `model` using the method in the keras' guide

the `model` can now do prediction. If you feed new image to it, it will tell you what is the wrtten number. 


---

## Deployed by NNoM

After the `model` is trained, the weights and parameters are already functional. We can now convert it to C language then being compiled in your MCU project. 

To conver the model, NNoM has provided an API `generate_model()`[API](api_nnom_utils.md) to automaticly do the job. Simply pass the `model` and the test dataset to it. It will do all the magics for you. 

~~~Python
generate_model(model, x_test, name='weights.h')
~~~

When the conversion is finished, you will find `weights.h` under your working folder. Simply copy the file to your MCU project, and call `model = nnom_model_create();` inside you `main()`. 

An C examples. 

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

Then your model is now running on you MCU. if you have support `printf` on your MCU, you should see the compiling info on your consoles. 

You can now use the model to predict your data. 

- Firstly, filling the input buffer `nnom_input_buffer[]` with your own data(image, signals) which is defined in `weights.h`. 
- Secondly, call `model_run(model);` to do your prediction. 
- Thirdly, read your result from `nnom_output_buffer[]`. The maximum number is the results. 

Please do check NNoM examples for more fancy methods. 

---

## What's More?

To be continue..

















