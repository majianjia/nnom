# Neural Network on Microcontroller (NNoM)

[![Build Status](https://travis-ci.org/majianjia/nnom.svg?branch=master)](https://travis-ci.org/majianjia/nnom)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

NNoM is a high-level inference Neural Network library specifically for microcontrollers. 

Document version 0.2.1

[[Chinese Intro]](rt-thread_guide.md)

**Highlights**

- Deploy Keras model to NNoM model with one line of code.
- User-friendly interfaces.
- Support complex structures; Inception, ResNet, DenseNet...
- High-performance backend selections.
- Onboard (MCU) evaluation tools; Runtime analysis, Top-k, Confusion matrix... 

The structure of NNoM is shown below:
![](figures/nnom_structure.png)

More detail avaialble in [Development Guide](guide_development.md)

## Licenses

NNoM is released under Apache License 2.0 since nnom-V0.2.0. 
License and copyright information can be found within the code.

---

## Why NNoM?
The aims of NNoM is to provide a light-weight, user-friendly and flexible interface for fast deploying.

Nowadays, neural networks are **wider**, **deeper**, and **denser**.
![](figures/nnom_wdd.png)
>[1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
>
>[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
>
>[3] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700-4708).


After 2014, the development of Neural Networks are more focus on structure optimising to improve efficiency and performance, which is more important to the small footprint platforms such as MCUs. 
However, the available NN libs for MCU are too low-level which make it sooooo difficult to use with these complex strucures. 

Therefore, we build NNoM to help embedded developers for faster and simpler deploying NN model directly to MCU. 
> NNoM will manage the strucutre, memory and everything else for developer. All you need is feeding your measurements then get the results. 

**NNoM is now working closely with Keras (You can easily learn [**Keras**](https://keras.io/) in 30 seconds!).**
There is no need to learn TensorFlow/Lite or other libs.  


---

## Documentations
API manuals are available within this site. 

**Guides**

[5 min to NNoM Guide](guide_5_min_to_nnom.md)

[Development Guide](guide_development.md)

[The temporary guide](A_Temporary_Guide_to_NNoM.md)

[Porting and optimising Guide](Porting_and_Optimisation_Guide.md) 

[RT-Thread Guide(Chinese)](rt-thread_guide.md)

[RT-Thread-MNIST example (Chinese)](example_mnist_simple_cn.md)

---

## Examples

**Documented examples**

Please check [examples](https://github.com/majianjia/nnom/tree/master/examples) and choose one to start with. 

Most recent Examples:

[MNIST-DenseNet example](https://github.com/majianjia/nnom/tree/master/examples/mnist-densenet)

[Octave Convolution](https://github.com/majianjia/nnom/tree/master/examples/octave-conv)

[Keyword Spotting](https://github.com/majianjia/nnom/tree/master/examples/keyword_spotting)

---


## Dependencies

NNoM now use the local pure C backend implementation by default. Thus, there is no special dependency needed. 

## Optimization
You can select [CMSIS-NN/DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN) as the backend for about 5x performance with ARM-Cortex-M4/7/33/35P. 

Check [Porting and optimising Guide](Porting_and_Optimisation_Guide.md) for detail. 


---
## Available Operations

> *Notes: NNoM now supports both HWC and CHW formats. Some operation might not support both format currently. Please check the tables for the current status. *

**Core Layers**

| Layers |HWC|CHW |Layer API|Comments|
| ------ |----|---- |------|------|
| Convolution  |✓|✓|Conv2D()|Support 1/2D|
| Depthwise Conv |✓|✓|DW_Conv2D()|Support 1/2D|
| Fully-connected |✓|✓| Dense()| |
| Lambda |✓|✓| Lambda() |single input / single output anonymous operation| 
| Batch Normalization |✓| ✓| N/A| This layer is merged to the last Conv by the script|
| Flatten|✓|✓| Flatten()| |
| SoftMax|✓|✓| SoftMax()| Softmax only has layer API| 
| Activation|✓|✓| Activation()|A layer instance for activation|
| Input/Output |✓|✓| Input()/Output()| |
| Up Sampling |✓| ✓|UpSample()||
| Zero Padding | ✓| ✓|ZeroPadding()||
| Cropping |✓ |✓ |Cropping()||

**RNN Layers**

| Layers | Status |Layer API|Comments|
| ------ | ------ | ------| ------|
| Recurrent NN | Under Dev.| RNN()| Under Developpment |
| Simple RNN | Under Dev. | SimpleCell()| Under Developpment |
| Gated Recurrent Network (GRU)| Under Dev. | GRUCell()| Under Developpment |

**Activations**

Activation can be used by itself as layer, or can be attached to the previous layer as ["actail"](docs/A_Temporary_Guide_to_NNoM.md#addictionlly-activation-apis) to reduce memory cost.

| Actrivation | HWC| CHW|Layer API|Activation API|Comments|
| ------ |-- |--|--|--|--|
| ReLU  | ✓|✓|ReLU()|act_relu()||
| TanH | ✓|✓|TanH()|act_tanh()||
|Sigmoid|✓|✓| Sigmoid()|act_sigmoid()||

**Pooling Layers**

| Pooling | HWC|CHW |Layer API|Comments|
| ------ |-- |--|--|--|
| Max Pooling  |✓|✓|MaxPool()||
| Average Pooling | ✓|✓|AvgPool()||
| Sum Pooling | ✓|✓|SumPool()| |
| Global Max Pooling |✓|✓|GlobalMaxPool()||
| Global Average Pooling |✓|✓|GlobalAvgPool()||
| Global Sum Pooling |✓|✓|GlobalSumPool()|A better alternative to Global average pooling in MCU before Softmax|

**Matrix Operations Layers**

| Matrix |HWC|CHW|Layer API|Comments|
| ------ |-- |--|--|--|
| Concatenate |✓|✓| Concat()| Concatenate through any axis|
| Multiple  |✓|✓|Mult()||
| Addition  |✓|✓|Add()||
| Substraction  |✓|✓|Sub()||
