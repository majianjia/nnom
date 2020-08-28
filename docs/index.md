# Neural Network on Microcontroller (NNoM)

[![Build Status](https://travis-ci.org/majianjia/nnom.svg?branch=master)](https://travis-ci.org/majianjia/nnom)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

NNoM is a high-level inference Neural Network library specifically for microcontrollers. 

Document version 0.4.1

[[中文简介]](rt-thread_guide.md)

**Highlights**

- Deploy Keras model to NNoM model with one line of code.
- Support complex structures; Inception, ResNet, DenseNet, Octave Convolution...
- User-friendly interfaces.
- High-performance backend selections.
- Onboard (MCU) evaluation tools; Runtime analysis, Top-k, Confusion matrix... 

The structure of NNoM is shown below:
![](figures/nnom_structure.png)

More detail avaialble in [Development Guide](guide_development.md)

## Latest Updates - v0.4.x

**Recurrent Layers (RNN) (0.4.1)**

Recurrent layers **(Simple RNN, GRU, LSTM)** are implemented in version 0.4.1. Support `statful` and `return_sequence` options. 

**New Structured Interface (0.4.0)** 

NNoM has provided a new layer interface called **Structured Interface**, all marked with `_s` suffix. which aims to use one C-structure to provided all the configuration for a layer. Different from the Layer API which is human friendly, this structured API are more machine friendly. 

**Per-Channel Quantisation (0.4.0)**

The new structred API supports per-channel quantisation (per-axis) and dilations for **Convolutional layers**. 

**New Scripts (0.4.0)**

From 0.4.0, NNoM will switch to structured interface as default to generate the model header `weights.h`. The scripts corresponding to structured interfaces are `nnom.py` while the Layer Interface corresponding to `nnom_utils.py`.

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

However, You will need to enable `libc` for dynamic memory allocation `malloc(), free(), and memset()`. Or you can port to the equivalent memory method in your system.  


## Optimization
You can select [CMSIS-NN/DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN) as the backend for about 5x performance with ARM-Cortex-M4/7/33/35P. 

Check [Porting and optimising Guide](Porting_and_Optimisation_Guide.md) for detail. 


---
## Available Operations

> *Notes: NNoM now supports both HWC and CHW formats. Some operation might not support both format currently. Please check the tables for the current status. *


**Core Layers**

| Layers | Struct API |Layer API|Comments|
| ------ |-------- |------|------|
| Convolution  |conv2d_s()|Conv2D()|Support 1/2D, support dilations (New!)|
| ConvTransposed (New!) |conv2d_trans_s()|Conv2DTrans()|Under Dev. |
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

| Layers | Status | Struct API |Comments|
| ------ | ------ | ------| ------|
| Recurrent NN Layer(New!) | Alpha | rnn_s()| Layer wrapper of RNN|
| Simple Cell (New!) | Alpha | simple_cell_s()||
| GRU Cell (New!) | Alpha | gru_cell_s()| Gated Recurrent Network |
| LSTM Cell (New!) | Alpha| lstm_s()| Long Short-Term Memory |

**Activations**

Activation can be used by itself as layer, or can be attached to the previous layer as ["actail"](docs/A_Temporary_Guide_to_NNoM.md#addictionlly-activation-apis) to reduce memory cost.

There is no structred API for activation currently, since activation are not usually used as a layer.

| Actrivation | Struct API |Layer API|Activation API|Comments|
| ------ |--|--|--|--|
| ReLU  | N/A |ReLU()|act_relu()||
| Leaky ReLU (New!) | N/A |LeakyReLU()|act_leaky_relu()||
| Adv ReLU(New!) | N/A |N/A|act_adv_relu()|advance ReLU, Slope, max, threshold|
| TanH | N/A |TanH()|act_tanh()||
| Hard TanH (New!)| N/A |TanH()||backend only|
|Sigmoid|N/A| Sigmoid()|act_sigmoid()||
|Hard Sigmoid (New!)|N/A| N/A| N/A|backend only|

**Pooling Layers**

| Pooling | Struct API|Layer API|Comments|
| ------ |--------|----|----|
| Max Pooling |maxpool_s()|MaxPool()||
| Average Pooling |avgpool_s()|AvgPool()||
| Sum Pooling |sumpool_s()|SumPool()||
| Global Max Pooling|global_maxpool_s()|GlobalMaxPool()||
| Global Average Pooling |global_avgpool_s()|GlobalAvgPool()||
| Global Sum Pooling |global_sumpool_s()|GlobalSumPool()|dynamic output shift|

**Matrix Operations Layers**

| Matrix |Struct API |Layer API|Comments|
| ------ |--|--|--|
| Concatenate |concat_s()| Concat()| Concatenate through any axis|
| Multiple  |mult_s()|Mult()||
| Addition  |add_s()|Add()||
| Substraction  |sub_s()|Sub()||


