# MNIST DenseNet Example 

**(Source code and models are under developing)**

They will be released shortly. 



This is an example of DenseNet classifier using [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/). 

## DenseNet
The detial of DenseNet is available in [the paper](https://arxiv.org/abs/1608.06993).

Which award the best paper in CVPR 2017 

The two typical graphs shows the basic ideas:

![Dense Net](https://github.com/majianjia/nnom/blob/master/examples/mnist-densenet/docs/densenet-1.png)
![Dense Block](https://github.com/majianjia/nnom/blob/master/examples/mnist-densenet/docs/densenet-2.jpeg)


Implementing the DenseNet using NNoM is quite straightforward. 

The codes below shows an exmaple of using 1 dense block.

~~~C
// the denseblock blocks takes the input layer, return the concatenated layer
nnom_layer_t * dense_block(nnom_model_t* model, nnom_layer_t * in)
{
	nnom_layer_t * x[5];
	
	x[0] = in;
	
	x[1] = model->hook(Conv2D(32, kernel(3, 3), stride(1, 1), PADDING_SAME, &c1_w, &c1_b), x[0]);
	x[1] = model->active(act_relu(), x[1]);
	// the above line can be replaced by one of the next 2 line. They logically perfrom the same
	// please check "actail" and activation API for more detail
	// x[1] = model->hook(ReLU(), x[1]); 
	// x[1] = model->hook(Activation(act_relu()), x[1]); 

	// concate 1,2 (Concate(-1), means concat by last axis, which is channel in HWC format)
	x[2] = model->mergex(Concat(-1), 2, x[0], x[1]); 
	x[2] = model->hook(Conv2D(32, kernel(3, 3), stride(1, 1), PADDING_SAME, &c2_w, &c2_b), x[2]);
	x[2] = model->active(act_relu(), x[2]);
	
	// concate 1,2,3
	x[3] = model->mergex(Concat(-1), 3, x[0], x[1], x[2]);
	x[3] = model->hook(Conv2D(32, kernel(3, 3), stride(1, 1), PADDING_SAME, &c3_w, &c3_b), x[3]);
	x[3] = model->active(act_relu(), x[3]);
	
	// concate 1,2,3,4
	x[4] = model->mergex(Concat(-1), 4, x[0], x[1], x[2], x[3]);
	x[4] = model->hook(Conv2D(32, kernel(3, 3), stride(1, 1), PADDING_SAME, &c4_w, &c4_b), x[4]);
	x[4] = model->active(act_relu(), x[4]);
	
	// concate 1,2,3,4,5 and return
	// the output channels should be input + 4 x 32
	return model->mergex(Concat(-1), 5, x[0], x[1], x[2], x[3], x[4]);
}
~~~
Now we can use the Dense Block to build a complete DenseNet
~~~C
int main()
{
	// input format
	input_layer = Input(shape(INPUT_HIGHT, INPUT_WIDTH, INPUT_CH), qformat(7, 0), nnom_input_data);
	
	// preprocessing
	x = model->hook(Conv2D(16, kernel(3, 3), stride(2, 2), PADDING_SAME, &c0_w, &c0_b),);
	
	// The dense block
	x = dense_block(&model, x);
	
	// the output of denseblock is 14 x 14 x 144(CH)
	// we need to reduce channels num to 10 for global average pooling
	x = model->hook(Conv2D(10, kernel(3, 3), stride(1, 1), PADDING_SAME, &c5_w, &c5_b),);

	// average pool
	x = model.hook(AvgPool(kernel(14, 14), stride(1, 1), PADDING_VALID), x);
	
	// output
	x = model.hook(Softmax(), x);
	x = model.hook(Output(shape(10,1,1), qformat(7, 0), nnom_output_data), x);
	
	// compile the model
	model_compile(&model, input_layer, x);
	// run it
	model_run(&model);
}

~~~


## Environments 
The example implementation is based on [RT-Thread](https://github.com/RT-Thread/rt-thread).

It uses Y-Modem, memory apis, consolse, ringbuffers which provided by RT-Thread

The targeted microcontroller is STM32L476


The detailed tutorial comes later. 
