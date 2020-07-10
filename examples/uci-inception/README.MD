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

# Preparation

You will need to download the dataset before doing the test. 

Linux user:
- run data/download_dataset.py to download the UCI HAR dataset

Windows user:
- download dataset from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip)
- unzip the file to this folder. Which will looks like `data/UCI HAR Dataset/...`

Copy all scripts from `nnom/scripts/*` into this folder. 



**Dependancy:**

In RT-Thread, you must open the support for Y-modem. 

