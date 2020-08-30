# RNN networks on UCI HAR dataset

This is an example of human activity recognition using [UCI HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones). 

This model stacks many RNN layer using the latest RNN layers supports by NNoM.


You can use both format like both below. 
~~~
    x = RNN(GRUCell(16), return_sequences=True)(x)
    x = GRU(16, return_sequences=True, stateful=True)(x)
~~~

- Option supports are `go_backwards`, `return_sequences` and `stateful`. Please leave other option as default. Please don't change the layer names/cell name. 
- Does not support `return_state`. But you can use `layer_callback()` to get it or set it. 
- For only Simple RNN, you can choose the activation between `sigmoid` and `tanh`. For GRU and LSTM, only default activation is support. i.e. `activation=sigmoid`, `recurrent_activation=tanh`
- All RNN Cell **doesnt not support** `linear` and `relu` activation. 

Run this example by using `python uci_rnn.py`. Require `scons` to build the c file. 

This is the same as the `auto-test example` which is used to test NNoM on Travis CI, but can also run on PC. 

# Preparation

You will need to download the dataset before doing the test. 

Linux user:
- run data/download_dataset.py to download the UCI HAR dataset

Windows user:
- download dataset from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip)
- unzip the file to this folder. Which will looks like `data/UCI HAR Dataset/...`

