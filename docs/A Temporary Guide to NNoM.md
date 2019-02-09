

# NNoM Structure

NNoM uses a layer-based structure. 

Layer is a container. Every operation (convolution, concat...) must be wrapped into a layer. 

A basic layer contains a list of **Input/Ouput modules** (I/O). Each of I/O contains a list of **Hook** (similar to Nodes in Keras).

**Hook**  stores the links to an I/O (other layer's)

**I/O** is a buffer to store input/output data of the operation. 

Dont be scared, check this:

![](https://github.com/majianjia/nnom/blob/master/docs/A%20Temporary%20Guide%20to%20NNoM/nnom_structures.png)

Next, we need APIs to create layers and build the model structures.  

# APIs
**layer APIs** and **construction APIs** are used to build a model. 

Layer APIs can create and return a new layer instance. Model APIs uses layer instances to build a model. 

**Layer APIs** such as `Conv2D(), Dense(), Activation()` ... which you can find in *nnom_layers.h*

**Construction APIs** such as `model.hook(), model.merge(), model.add()` ... which you can find in `new_model()` at *nnom.c*


For example, to add a convolution layer into sequencial model:
~~~c
model.add(&model, Conv2D(16, kernel(1, 9), stride(1, 2), PADDING_SAME, &c1_w, &c1_b));
~~~



In functional model, the links between layer is specified explicitly by using `model.hook()`
~~~c
x = model.hook(Conv2D(16, kernel(1, 9), stride(1, 2), PADDING_SAME, &c1_w, &c1_b), input_layer);
x = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x);
~~~


## Construction APIs
Construction APIs are statics functions located in nnom.c
Currently are:

Sequencial Construction API 
~~~c
nnom_status_t model.add(nnom_model_t* model,  nnom_layer_t *layer);
~~~

Functional Construction API

~~~c
// hook the current layer to the input layer
// this function only to connect (single output layer) to (single input layer). 
// return the curr (layer) instance
nnom_layer_t * model.hook(nnom_layer_t* curr, nnom_layer_t *last)
~~~

~~~c
// merge 2 layer's output to one output by provided merging method(a mutiple input layer)
// method = merging layer such as (concat(), dot(), mult(), add())
// return the method (layer) instance
nnom_layer_t * model.merge(nnom_layer_t *method, nnom_layer_t *in1, nnom_layer_t *in2)
~~~

~~~c
// Same as model.merge()
// Except it can take mutiple layers as input. 
// num = the number of layer
// method: same as model.merge()
nnom_layer_t * model.mergex(nnom_layer_t *method, int num, ...)
~~~

~~~c
// This api will merge the activation to the targeted layerto reduce an extra activation layer
// activation such as (act_relu(), act_tanh()...)
nnom_layer_t * model.active(nnom_activation_t* act, nnom_layer_t * target)
~~~
For `model.active()`, please check Activation APIs below. 



## Layer APIs

Layers APIs are listed in *nnom_layers.h*

Input/output layers are neccessary for a model. They are responsible to copy data from user's input buffer, and copy out to user's output buffer. 
~~~c
// Layer APIs 
// input/output
nnom_layer_t* Input(nnom_shape_t input_shape, nnom_qformat_t fmt, void* p_buf);
nnom_layer_t* Output(nnom_shape_t output_shape, nnom_qformat_t fmt, void* p_buf);
~~~

Pooling as they are
~~~c
// Pooling, kernel, strides, padding
nnom_layer_t* MaxPool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad);
nnom_layer_t* AvgPool(nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad);
~~~

Activation's **Layers API** are started with capital letter. They are differed from the **Activation API**, which start with `act_*` and retrun an activation instance.
Pleas check the Activation APIs below for more detail. 

They return a **layer** instance. 
~~~c
// Activation layers take activation instance as input.  
nnom_layer_t* Activation(nnom_activation_t *act);		
// Activation's layer API. 
nnom_layer_t* ReLU(void);
nnom_layer_t* Softmax(void);
nnom_layer_t* Sigmoid(void);
nnom_layer_t* TanH(void);
~~~

Matrix API. 

These layers normally take 2 or more layer's output as their inputs. 

They also called "merging method", which must be used by `model.merge(method, in1, in2)`or `model.mergex(method, num of input, in1, in2, 1n3 ...)`
~~~c
// Matrix
nnom_layer_t* Add(void);
nnom_layer_t* Sub(void);
nnom_layer_t* Mult(void);
nnom_layer_t* Concat(int8_t axis);
~~~

Flatten change the shapes to (x, 1, 1)
~~~c
// utils
nnom_layer_t* Flatten(void);
~~~

Stable NN layers.
For more developing layers, please check the source codes. 

~~~c
// conv2d
nnom_layer_t* Conv2D(uint32_t filters, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad,
	nnom_weight_t *w, nnom_bias_t *b);

// depthwise_convolution
nnom_layer_t* DW_Conv2D(uint32_t multiplier, nnom_shape_t k, nnom_shape_t s, nnom_padding_t pad, 
	nnom_weight_t *w, nnom_bias_t *b);

// fully connected, dense
nnom_layer_t* Dense(size_t output_unit, nnom_weight_t *w, nnom_bias_t *b);
~~~

About the missing **Batch Normalization Layer**

Batch Normalization layer can be fused into the last convolution layer. So NNoM currently does not provide a Batch Normalization Layer. It might be implemented as a single layer in the future. However, currently, please fused it to the last layer.

[Further reading about fusing BN parameters to conv weights](https://tkv.io/posts/fusing-batchnorm-and-conv/)

[Fusing batch-norm layers](https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Quant_guide.md#fusing-batch-norm-layers)


## Addictionlly, Activation APIs
 
Actication APIs are not essential in the original idea. The original idea is making eveything as a layer. 

However, single layer instances cost huge amount of memories(100~150 Bytes), while activations are relativly simple, mostly have same input/output shape, a few/none parameter(s)...

Therefore, to reduce the complexity, the "actail"(activation tail) is added to each layer instance. If a layer's Actail is not null, it will be called right after the layer is executed. Actail takes activation instance as input. The model API, `model.active()` will attach the activation to the layer's actail. 

~~~c
// attach act to target_layer, return the target layer instance.
nnom_layer_t * model.active(nnom_activation_t* act, nnom_layer_t * target_layer)
~~~
 
The Activation APIs are listed in *nnom_activations.h*

~~~c
// Activation
nnom_activation_t* act_relu(void);
nnom_activation_t* act_softmax(void);
nnom_activation_t* act_sigmoid(void);
nnom_activation_t* act_tanh(void);
~~~
 
 
## Model API

A model instance contains the starting layer, the end layer and other neccessary info. 


~~~c
// Create or initial a new model() 
nnom_model_t* 	new_model(nnom_model_t* m);

// Delete the model. This is not functional currently. 
void model_delete(nnom_model_t* m);  

// Compile a sequencial model. 
nnom_status_t 	sequencial_compile(nnom_model_t *m);

// Compile a functional model with specified input layer and output layer. 
// if output = NULL, the output is automatic selected. 
nnom_status_t 	model_compile(nnom_model_t *m, nnom_layer_t* input, nnom_layer_t* output);

// Run the model.
nnom_status_t 	model_run(nnom_model_t *m);
~~~
 

# Evaluation

The evaluation methods are listed in `nnom_utils.h`

They run the model with testing data, then evaluate the model. Includes Top-k accuracy, confusion matrix, runtime stat...

Please refer to [UCI HAR example](https://github.com/majianjia/nnom/tree/master/examples/uci-inception) for usage. 
~~~c
// create a prediction
// input model, the buf pointer to the softwmax output (Temporary, this can be extract from model)
// the size of softmax output (the num of lable)
// the top k that wants to record. 
nnom_predic_t* prediction_create(nnom_model_t* m, int8_t* buf_prediction, size_t label_num, size_t top_k_size);// currently int8_t 

// after a new data is set in input
// feed data to prediction
// input the current label, (range from 0 to total number of label -1)
// (the current input data should be set by user manully to the input buffer of the model.)
uint32_t prediction_run(nnom_predic_t* pre, uint32_t label);

// to mark prediction finished
void prediction_end(nnom_predic_t* pre);

// free all resources
void predicetion_delete(nnom_predic_t* pre);

// print matrix
void prediction_matrix(nnom_predic_t* pre);

// this function is to print sumarry 
void prediction_summary(nnom_predic_t* pre);

// -------------------------------

// stand alone prediction API
// this api test one set of data, return the prediction 
// input the model's input and output bufer
// return the predicted label
uint32_t nnom_predic_one(nnom_model_t* m, int8_t* input, int8_t* output); // currently int8_t 

// print last runtime stat of the model
void model_stat(nnom_model_t *m);
~~~


## Demo of Evaluation

The UCI HAR example runs on RT-Thread, uses Y-Modem to receive testing dataset, uses ringbuffer to store data, and the console (msh) to print the results. 

The layer order, activation, output shape, operation, memory of I/O, and assigned memory block are shown. 
It also summarised the memory cost by neural network. 
![Model Compiling](https://github.com/majianjia/nnom/blob/master/docs/gifs/nnom_compile.gif)

Type `predic`, then use Y-Modem to send the data file. The model will run once enough data is received.
![Start Prediction](https://github.com/majianjia/nnom/blob/master/docs/gifs/nnom_predic_start.gif)

When the file copying done, the runtime summary, Top-k and confusion matrix will be printed
![Prediction finished](https://github.com/majianjia/nnom/blob/master/docs/gifs/nnom_predic_finished.gif)

Optionally, the runtime stat detail of each layer can be printed by `nn_stat`
![Print stat](https://github.com/majianjia/nnom/blob/master/docs/gifs/nnom_stat.gif)
 
PS: The "runtime stat" in the animation is not correct, due to the test chip is overclocking (STM32L476 @ 160MHz, 2x overclocking), and the timer is overclocking as well. 

However, the numbers in prediction summary are correct, because they are measured by system_tick timer which is not overclocking. 
 











