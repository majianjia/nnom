# Development Guide

> This guide is to give a detail instruction to NNoM.
> 

## Introduction

### What is NNoM different from others?

NNoM is a higher level inference framework. The most obvious feature is the human understandable interface. 

It is also a layer-based framework, instead of a operator-based. A layer might contains a few operators. 

It natively supports complex model structure. High-efficency network always benefited from complex structure.

It provide layer-to-layer analysis to help developer optimize their models. 

### Develop ad-hoc model vs. use pre-trained models. 

The famous pre-trained models are more for the image processing side. 
They are effecient on such mobile phones. 
But they are still too buckly if the MCU didn't fit with at least 250K RAM and a hardware Neural Network Accelorator. 

>  MobileNet  V1  model  with  depth  multi-plier  (0.25x) ...  STM32 F746 ... CMSIS-NN kernels to program the depthwise and pointwise convolutions ... approximately 0.75 frames/sec 
Source: [Visual Wake Words Dataset](https://arxiv.org/abs/1906.05721)

However, MCUs should not really do the image processing. The data they normally process are normally not visual but other time sequance measurement. 
For example, the accelerometer data consist of 3 axis (channel) measurement per timestamp. 

Dealing with these data, building the ad-hoc models for each application is the only option. 

Building an ad-hoc model is sooo easy with NNoM since most of the codes are automative generated. 


### What can NNoM provide to embedded engineers?
It provide an **easy to use** and **easy to evaluate** inference tools for fast neural network development. 

As embedded engineers, we might not know well how does neural network work and how can we optimize it for the MCU. 

NNoM together with Keras can help you to start practicing within half an hour. There is no need to learn other ML lib from scratch. Deployment can be done with one line of python code after you have trained an model using Keras. 

Other than building a model, NNoM also provides a set of evaluation methods. These evaluation methods will give the developer a layer-to-layer performance evaluation of the model. 

Developers can then modify the ad-hoc model to increase efficency or to lower the memory cost. 
(Please check the following Performance sections for detials.)

## NNoM Structure

As mentioned in many other docs, NNoM uses a layer-based structure. 
The most benefit is the model structure can be directly seemed from the codes.

It also makes the model convertion from other layer-based lib (Keras, TensorLayer, Caffe) to NNoM model very straight forward. 

NNoM uses a compiler to manage the layer structure and other resources. After compiling, all layers inside the model will be put into a shortcut list per the running order. Beside that, arguments will be filled in and the memory will be allocated to each layer (Memory are reused in between layers). Therefore, no memory allocation performed in the runtime, performance is the same as running backend function directly.   

The NNoM is more on manage the higher-level structure, context argument and memory. The actual arithmetics are done by the backend functions.

Currently, NNoM support a pure C backend and CMSIS-NN backend. 
The CMSIS-NN is an highly optimized low-level NN core for ARM-Cortex-M microcontroller. 
Please check the [optimize guide](Porting_and_Optimisation_Guide.md) for utilisation.

## Optimation 

The CMSIS-NN can provide upto 5 times performance comparing to the pure C backend on Cortex-M MCUs. It maximises the performance by using SIMD and other instructions(__SSAT, ...).

These optimizations come with different constrains. This is why CMSIS-NN provides many variances to one operators (such as 1x1 convolution, RGB convolution, none-square/square, they are all convolution only with different routines). 

NNoM will automaticly select the best operator for the layer when it is available. Sometime, it is not possible to use CMSIS-NN because the condition is not met. CMSIS-NN provides a subset operator to the local pure C backend. When it is not possible to use CMSIS-NN, NNoM will run the layer using the C backend end instead. It is vary from layer to layer whether use CMSIS-NN or C backend. 

The example condition for convolutions are list below:

|Operation|Input Channel|Output Channel|
|-------|-------|-------|
|Convolution |  multiple of 4| multiple of 2 |
|Pointwise Convolution |  multiple of 4| multiple of 2 |
|Depthwise Convolution |  multiple of 2| multiple of 2 |

The full details can be found in [CMSIS-NN's source code](https://github.com/ARM-software/CMSIS_5) and [documentation](https://www.keil.com/pack/doc/CMSIS/NN/html/index.html). 
Some of them can be further optimized by square shape, however, the optimization is less significant. 

> Trick, if you keep the channel size is a multiple of 4, it should work in most of the case.

If you are not sure whether the optimization is working, simply us the `model_stat()` in [Evaluation API](api_evaluation.md) to print the performance of each layer. Comparison will be shown in the following sections.  

Fully connected layers and pooling layers are less constrained. 

## Performance

Performances are vary. 
Efficiencies are more constant.

We can use *Multiply–accumulate operation (MAC) per Hz (MACops/Hz)* to evaluate the efficency. 
It simply means how many MAC can be done in one cycle. 

Currently, NNoM only count MAC operations on Convolution layers and Dense layers since other layers (pooling, padding) are much lesser. 

Running an model on CMSIS-NN and NNoM will have the same performance, when a model is fully compliant  with CMSIS-NN and running on Cortex-M4/7/33/35P. ("compliant" means it meets the optimization condition in above discussion). 

For example, in [CMSIS-NN paper](https://arxiv.org/pdf/1801.06601), the authors used an STM32F746@216MHz to run a model with 24.7M(MACops) tooks 99.1ms in total.

The runtime of each layer were recorded. What hasn't been shown in the paper is this table. (refer to Table 1 in the paper)

|       |Layer|Input ch|output ch|Ops|Runtime|Efficiency|
|-------|-------|-------|-----|-------|----|-------|
|Layer 1| Convolution|3|32|4.9M|31.4ms|	|
|Layer 3| Convolution|32|32|13.1M|42.8ms||
|Layer 5| Convolution|32|64|6.6M|22.6ms||
|Layer 7| Fully-connected|1024|10|20k|0.1ms||
|Total| |||24.7M|99.1||

 


## Evaluations

Evaluation is equally important to building the model. 

~~~
Start compiling model...
Layer(#)         Activation    output shape    ops(MAC)   mem(in, out, buf)      mem blk lifetime
-------------------------------------------------------------------------------------------------
#1   Input      -          - (  28,  28,   1)          (   784,   784,     0)    1 - - -  - - - - 
#2   Conv2D     - ReLU     - (  28,  28,  12)      84k (   784,  9408,    36)    1 1 1 -  - - - - 
#3   MaxPool    -          - (  14,  14,  12)          (  9408,  2352,     0)    1 1 1 -  - - - - 
#4   Cropping   -          - (  11,   7,  12)          (  2352,   924,     0)    1 1 - -  - - - - 
#5   Conv2D     - ReLU     - (  11,   7,  24)     199k (   924,  1848,   432)    1 1 1 -  - - - - 
#6   MaxPool    -          - (   3,   2,  24)          (  1848,   144,     0)    1 1 1 -  - - - - 
#7   ZeroPad    -          - (   6,   9,  24)          (   144,  1296,     0)    1 1 - -  - - - - 
#8   Conv2D     - ReLU     - (   6,   9,  24)     279k (  1296,  1296,   864)    1 1 1 -  - - - - 
#9   UpSample   -          - (  12,  18,  24)          (  1296,  5184,     0)    1 - 1 -  - - - - 
#10  Conv2D     - ReLU     - (  12,  18,  48)    2.23M (  5184, 10368,   864)    1 1 1 -  - - - - 
#11  MaxPool    -          - (   6,   9,  48)          ( 10368,  2592,     0)    1 1 1 -  - - - - 
#12  Dense      - ReLU     - (  64,   1,   1)     165k (  2592,    64,  5184)    1 1 1 -  - - - - 
#13  Dense      -          - (  10,   1,   1)      640 (    64,    10,   128)    1 1 1 -  - - - - 
#14  Softmax    -          - (  10,   1,   1)          (    10,    10,     0)    1 1 - -  - - - - 
#15  Output     -          - (  10,   1,   1)          (    10,    10,     0)    1 - - -  - - - - 
-------------------------------------------------------------------------------------------------
Memory cost by each block:
 blk_0:5184  blk_1:2592  blk_2:10368  blk_3:0  blk_4:0  blk_5:0  blk_6:0  blk_7:0  
 Total memory cost by network buffers: 18144 bytes
Compling done in 179 ms

~~~

~~~
Print running stat..
Layer(#)        -   Time(us)     ops(MACs)   ops/us 
--------------------------------------------------------
#1        Input -        11                  
#2       Conv2D -      5848          84k     14.47
#3      MaxPool -       698                  
#4     Cropping -        16                  
#5       Conv2D -      3367         199k     59.27
#6      MaxPool -       346                  
#7      ZeroPad -        36                  
#8       Conv2D -      4400         279k     63.62
#9     UpSample -       116                  
#10      Conv2D -     33563        2.23M     66.72
#11     MaxPool -      2137                  
#12       Dense -      2881         165k     57.58
#13       Dense -        16          640     40.00
#14     Softmax -         3                  
#15      Output -         1                  

Summary:
Total ops (MAC): 2970208(2.97M)
Prediction time :53439us
Efficiency 55.58 ops/us
NNOM: Total Mem: 20236

~~~


## Others

### Memeory management in NNoM

### 

