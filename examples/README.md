
## What we have

The list below are the remiders for each example. 
For details, please see the docs under the folder individually. 

- **auto_test** is a PC based example, which is use by NNoM for travis CI. However, it is a very good demo for NNoM without borthering MCU. 
You can run it directly on your PC. 
- **keyword_spotting** is a regular convolution model to spot speech commands train by [google speech dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). 
This example use MFCC to extract the voice features, then use neural network to classified these commands into 30+ classes. 
The model has achieved around 90% Top 1 accuracy. 
- **mnist-simple** is an interactive example for RT-Thread using its Mesh shell. 
This example come with 10 embedded images allows users to do prediction through terminal. [Chinese guides available](../docs/example_mnist_simple_cn.md).
- **octave-conv** is to show how to construct the latest Octave convolution in Keras then deploy to NNoM. 
- **uci-inception** is an example using data from motion sensors and Inception structure. 
It is an interactive example using shell and Y-modem to transmit testing data. 
- **mnist-cnn** is an entry level example using jupyter notebook following this [Keras tutorial](https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/) 
- **mnist-densenet** is a example showing how to use DenseNet with NNoM. 
This example can be compiled in PC using scons. It is also been use for Travis CI. Please check the Travis logging as well. 

## Recommendations 

If you want to try it first on PC, start with **auto_test** and follow the guide in it. 

If you are completely void in ML or Neural network, start with **[mnist-cnn](mnist-cnn)** and the external tutorial for Keras. 

If you are trying to handle time sequence data (e.g. sensor measurement or voice), please check **[keyword_spotting](keyword_spotting)** and **[uci-inception](uci-inception)**.

If you are using RT-Thread, the very first example you should try is **[mnist-simple](mnist-simple)**, just compile and run. 


## Environment 

All example require `Python3, Keras, Tensorflow` installed in your PC. 

**mnist-cnn** and **mnist-densenet** are not relied on RT-Thread. 

Others require [RT-Thread] and sometimes its Y-modem components. 

**keyword_spotting** require a microphone on your development board and the porting of mic, if you are not using [32L476GDISCOVERY](https://www.st.com/en/evaluation-tools/32l476gdiscovery.html)



