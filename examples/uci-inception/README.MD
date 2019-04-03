# uci-inception example

This is an example of human activity recognition using [UCI HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones). 

The model was built with Inception structures. 

~~~
      Input
        |
        |
      Conv
    /   |   \
   /    |    \
Conv  Conv  Maxpool
   \    |    /
    \   |   /
     Concat
        |
      Conv
        |   
      Dense
        |
      Dense
        |
     Softmax
~~~

The example implementation is based on [RT-THREAD](https://github.com/RT-Thread/rt-thread).

Please follow the guide [doc preparing]

**Dependancy:**

In RT-Thread, you must open the support for Y-modem. 

