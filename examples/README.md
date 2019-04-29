
## Examples

The list below are remiders for each example. 
For details of that example, please check the docs under the folder individually. 

- **keyword_spotting** is a regular convolution model to spot speech commands train by google speech dataset. 
This example use MFCC to extract the voice features, then use NN to classified these commands into 30+ class. 
Has achieved around 90% Top 1 accuracy. 
- **mnist-cnn** is a entry level example using jupyter notebook following this [Keras tutorial](https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/) 
- **mnist-densenet** is a example showing how to use DenseNet with NNoM. 
This example can be compiled in PC using scons. It is also been use for Travis CI. Please check the Travis logging as well. 
- **mnist-simple** is an interactive example for RT-Thread using its Mesh shell. 
This example come with 10 embedded images allows users to do prediction through terminal. [Chinese guides available](../docs/example_mnist_simple_cn.md).
- **octave-conv** is to show how to construct the latest Octave convolution in Keras then deploy to NNoM. 
- **uci-inception** is an example using data from motion sensors and Inception structure. 
It is an interactive example using shell and Y-modem to transmit testing data. 


## Environment 

All example require `Python3, Keras, Tensorflow` installed in your PC. 

**mnist-cnn** and **mnist-densenet** are not relied on RT-Thread. 

Others require [RT-Thread] and sometimes its Y-modem components. 

