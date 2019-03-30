# Neural Network on Microcontroller (NNoM)

NNoM is a higher-level layer-based Neural Network library specifically for microcontrollers. 

**Highlights**

- Deploy Keras model to NNoM model with one line of code.
- User-friendly interfaces.
- Support complex structures; Inception, ResNet, DenseNet...
- High-performance backend selections.
- Onboard (MCU) evaluation tools; Runtime analysis, Top-k, Confusion matrix... 

Guides:

[RT-Thread Guide(中文指南)](rt-thread_guide.md)

[The temporary guide](A_Temporary_Guide_to_NNoM.md)

[Porting and optimising Guide](Porting_and_Optimisation_Guide.md)

Examples:

[RT-Thread-MNIST example (中文)](example_mnist_simple_cn.md)

[MNIST-DenseNet example](https://github.com/majianjia/nnom/tree/master/examples/mnist-densenet)

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
Therefore, we build NNoM to help developers to manage the structures, memory and parameters, even with the automatic tools for faster and simpler deploying.

Now with NNoM, you are free to play with these more up-to-date, decent and complex structures on MCU.

With [**Keras**](https://keras.io/) and our tools, deploying a model only takes a few line of codes.

Please do check the [examples](https://github.com/majianjia/nnom/tree/master/examples).

---

## Dependencies

NNoM now use the local pure C backend implementation by default. Thus, there is no special dependency needed. 

## Optimization
You can select [CMSIS-NN/DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN) as the backend for about 5x performance with ARM-Cortex-M4/7/33/35P. 

Check [Porting and optimising Guide](Porting_and_Optimisation_Guide.md) for detail. 
