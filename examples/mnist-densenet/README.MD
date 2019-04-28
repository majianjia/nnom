# MNIST DenseNet Example 

This is an example of DenseNet classifier using [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/). 

## DenseNet

The detial of DenseNet is available in [the paper](https://arxiv.org/abs/1608.06993).

Which has awarded the best paper in CVPR 2017 

The two graphs show the basic idea of DenseNet:

![Dense Net](https://github.com/majianjia/nnom/blob/master/examples/mnist-densenet/docs/densenet-1.png)
![Dense Block](https://github.com/majianjia/nnom/blob/master/examples/mnist-densenet/docs/densenet-2.jpeg)

Implementation of the DenseNet using NNoM is quite straightforward. 

The codes in the example uses the automatic tools `generate_model(model, x_test[:10], name=weights)` to deploy the Keras DenseNet model into `weights.h`. Please see the python script (Keras model) for details. 

Then the function in C file `model = nnom_model_create();` loads and compiles the model in the MCU side. 

## Build it manually

Building it manually is recommended if you want to gain more control or if you want to try some new networks which the automatic tools have yet supported. 

The code below shows an exmaple of building 1 dense block.

~~~C
// the denseblock blocks takes the input layer, return the concatenated layer
nnom_layer_t * dense_block(nnom_model_t* model, nnom_layer_t * in)
{
	nnom_layer_t * x[5];
	
	x[0] = in;
	
	x[1] = model->hook(Conv2D(32, kernel(3, 3), stride(1, 1), PADDING_SAME, &c1_w, &c1_b), x[0]);
	x[1] = model->active(act_relu(), x[1]);

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
	x = model.hook(Conv2D(16, kernel(3, 3), stride(2, 2), PADDING_SAME, &c0_w, &c0_b),);
	
	// The dense block
	x = dense_block(&model, x);
	
	// the output of denseblock is 14 x 14 x 144(CH)
	// we need to reduce channels num to 10 for global average pooling
	x = model.hook(Conv2D(10, kernel(3, 3), stride(1, 1), PADDING_SAME, &c5_w, &c5_b),);

	// global average pool
	x = model.hook(GlobalAvgPool()), x);
	
	// output
	x = model.hook(Flatten(), x)
	x = model.hook(Softmax(), x);
	x = model.hook(Output(shape(10,1,1), qformat(7, 0), nnom_output_data), x);
	
	// compile the model
	model_compile(&model, input_layer, x);
	// run it
	model_run(&model);
}

~~~

