
# Neural Network on Microcontroller (NNoM)
[![Build Status](https://travis-ci.org/majianjia/nnom.svg?branch=master)](https://travis-ci.org/majianjia/nnom)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

NNoM is a high-level inference Neural Network library specifically for microcontrollers. 

[[English Manual]](https://majianjia.github.io/nnom/) [[Chinese Intro]](docs/rt-thread_guide.md) 

**Highlights**

- Deploy Keras model to NNoM model with one line of code.
- Support complex structures; Inception, ResNet, DenseNet, Octave Convolution...
- User-friendly interfaces.
- High-performance backend selections.
- Onboard (MCU) evaluation tools; Runtime analysis, Top-k, Confusion matrix... 

The structure of NNoM is shown below:
![](docs/figures/nnom_structure.png)

More detail avaialble in [Development Guide](docs/guide_development.md)

Discussions welcome using [issues](https://github.com/majianjia/nnom/issues). 
Pull request welcome. QQ/TIM group: 763089399.

## Latest Updates - v0.4.x

**New Structured Interface**

NNoM has provided a new layer interface called **Structured Interface**, all marked with `_s` suffix. which aims to use one C-structure to provided all the configuration for a layer. Different from the Layer API which is human friendly, this structured API are more machine friendly. 

**Per-Channel Quantisation**

The new structred API supports per-channel quantisation (per-axis) and dilations for Convolutional layers. 


**New Scripts**

From 0.4.0, NNoM will switch to structured interface as default to generate the model header `weights.h`. The scripts corresponding to structured interfaces are `nnom.py` while the Layer Interface corresponding to `nnom_utils.py`.

## Licenses

NNoM is released under Apache License 2.0 since nnom-V0.2.0. 
License and copyright information can be found within the code.

## Why NNoM?
The aims of NNoM is to provide a light-weight, user-friendly and flexible interface for fast deploying on MCU.

Nowadays, neural networks are **wider**, **deeper**, and **denser**.
![](docs/figures/nnom_wdd.png)
>[1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
>
>[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
>
>[3] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700-4708).

After 2014, the development of Neural Networks are more focus on structure optimising to improve efficiency and performance, which is more important to the small footprint platforms such as MCUs. 
However, the available NN libs for MCU are too low-level which make it sooooo difficult to use with these complex strucures. 

Therefore, we build NNoM to help embedded developers for faster and simpler deploying NN model directly to MCU. 
> NNoM will manage the strucutre, memory and everything else for the developer. All you need to do is feeding your new measurements and getting the results. 

**NNoM is now working closely with Keras (You can easily learn [**Keras**](https://keras.io/) in 30 seconds!).**
There is no need to learn TensorFlow/Lite or other libs.  


## Documentations

**Guides**

[5 min to NNoM Guide](docs/guide_5_min_to_nnom.md)

[The temporary guide](docs/A_Temporary_Guide_to_NNoM.md)

[Porting and optimising Guide](docs/Porting_and_Optimisation_Guide.md) 

[RT-Thread Guide(Chinese)](https://majianjia.github.io/nnom/rt-thread_guide/)

[RT-Thread-MNIST example (Chinese)](docs/example_mnist_simple_cn.md)

## Examples

**Documented examples**

Please check [examples](https://github.com/majianjia/nnom/tree/master/examples) and choose one to start with. 

## Available Operations

[[API Manual]](https://majianjia.github.io/nnom/)

> *Notes: NNoM now supports both HWC and CHW formats. Some operation might not support both format currently. Please check the tables for the current status. *



**Core Layers**

| Layers | Struct API |Layer API|Comments|
| ------ |----|---- |------|------|
| Convolution  |conv2d_s()|Conv2D()|Support 1/2D, support dilations (New!)|
| ConvTransposed  |conv2d_trans_s()|Conv2DTrans()|Under Dev. (New!)|
| Depthwise Conv |dwconv2d_s()|DW_Conv2D()|Support 1/2D|
| Fully-connected |dense_s()| Dense()| |
| Lambda |lambda_s()| Lambda() |single input / single output anonymous operation| 
| Batch Normalization |N/A| N/A| This layer is merged to the last Conv by the script|
| Flatten|flatten_s()| Flatten()| |
| SoftMax|softmax_s()| SoftMax()| Softmax only has layer API| 
| Activation|N/A| Activation()|A layer instance for activation|
| Input/Output |input_s()/output_s()| Input()/Output()| |
| Up Sampling |upsample_s()|UpSample()||
| Zero Padding | zeropadding_s()|ZeroPadding()||
| Cropping |cropping_s() |Cropping()||

**RNN Layers**

| Layers | Status |Layer API|Comments|
| ------ | ------ | ------| ------|
| Recurrent NN | Under Dev.| RNN()| Under Developpment |
| Simple RNN | Under Dev. | SimpleCell()| Under Developpment |
| Gated Recurrent Network (GRU)| Under Dev. | GRUCell()| Under Developpment |

**Activations**

Activation can be used by itself as layer, or can be attached to the previous layer as ["actail"](docs/A_Temporary_Guide_to_NNoM.md#addictionlly-activation-apis) to reduce memory cost.

There is no structred API for activation currently, since activation are not usually used as a layer.

| Actrivation | Struct API |Layer API|Activation API|Comments|
| ------ |--|--|--|--|
| ReLU  | N/A |ReLU()|act_relu()||
| Leaky ReLU (New!) | N/A |ReLU()|act_relu()||
| TanH | N/A |TanH()|act_tanh()||
|Sigmoid|N/A| Sigmoid()|act_sigmoid()||

**Pooling Layers**

| Pooling | Struct API|Layer API|Comments|
| ------ |----|----|----|----|
| Max Pooling |maxpool_s()|MaxPool()||
| Average Pooling |avgpool_s()|AvgPool()||
| Sum Pooling |sumpool_s()|SumPool()||
| Global Max Pooling|global_maxpool_s()|GlobalMaxPool()||
| Global Average Pooling |global_avgpool_s()|GlobalAvgPool()||
| Global Sum Pooling |global_sumpool_s()|GlobalSumPool()|A better alternative to Global average pooling in MCU before Softmax|

**Matrix Operations Layers**

| Matrix |Struct API |Layer API|Comments|
| ------ |--|--|--|
| Concatenate |concat_s()| Concat()| Concatenate through any axis|
| Multiple  |mult_s()|Mult()||
| Addition  |add_s()|Add()||
| Substraction  |sub_s()|Sub()||


## Dependencies

NNoM now use the local pure C backend implementation by default. Thus, there is no special dependency needed. 

To use Log functions, you will need to enable libc in your projects. 


## Optimization
[CMSIS-NN/DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN) is an optimized backend for ARM-Cortex-M4/7/33/35P. You can select it for up to 5x performance compared to the default C backend. NNoM will use the equivalent method in CMSIS-NN if the condition met. 

Please check [Porting and optimising Guide](docs/Porting_and_Optimisation_Guide.md) for detail. 

## Known Issues
### Converter do not support implicitly defined activations
The script currently does not support implicit act:
~~~
Dense(32, activation="relu")
~~~
Use the explicit activation instead. 
~~~
Dense(32)
Relu()
~~~

## Contacts
Jianjia Ma

majianjia@live.com

## Citation Required
Please contact us using above details. 

