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
It's **easy to use** and **easy to evaluate**. 

As embedded engineers, we might not know well how does neural network work and how can we optimize it for the MCU. 

NNoM together with Keras can help you to start practicing within half an hour. There is no need to learn other ML lib from scratch. Deployment can be done with one line of python code following Keras model. 

Other than building a model, NNoM also provides a set of evaluation methods. Evaluation will give developer a layer-to-layer performance evaluation to user. 

Developers then can modify the ad-hoc model to increase efficency or to lower the memory cost. 
(Please check the following Performance sections for detials.)


## NNoM Structure


## Optimation 


## Performance


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

