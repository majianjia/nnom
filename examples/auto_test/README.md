
# Auto-test Example

[A guide on medium is available](https://medium.com/@majianjia1991/running-a-complex-neural-network-in-microcontroller-fd1c5fbf30a5?source=friends_link&sk=06c867efdb37ea21f47192dde9e1f2cc)

# Try it on PC

This **auto-testing example** is a perfect starting point without bothering MCU. It is used for automatic test on Travis CI, so it can run on your PC already. This example will do a simple classification on a famous 10-digit handwriting [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Environments

You will need `Python 2/3`, `SCons`, `Keras`, and `a C language compiler (MSVC for Windows or GCC for Linux)`

With **Linux**, you can install them using the following scripts.

~~~
sudo apt-get install scons 
sudo pip install tensorflow keras matplotlib scikit-learn
~~~

For **Windows**, I recommend installing [Anaconda](https://www.anaconda.com/). Then use the below script to install Keras and scons using “conda”. You probably need to install MSVC compiler by installing [Visual Studio and select the C++ supports](https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=vs-2019).
~~~
conda install keras
conda install scons
~~~

Then download the NNoM codes. You can download the codes from NNoM GitHub repo in any method you like.

## Let’s run it!

It is simple since there is only one single Python file, `main.py`. Just run it.

**Windows**: You will need a CMD or PowerShell to run the python codes. The easiest way is open `nnom/example/auto_test` with File Explorer. Right-click the empty space with pressing “Shift” on your keyboard. Then, select "open PowerShell windows here".

Now you can run the example in the PowerShell windows by the below script. Same for **Linux** user.(if you prefer to run the python code on an IDE, you are free to do so)

~~~
python main.py
~~~

The Python script will 
- Build and train the Keras model。
- Generate the NNoM model (weights.h) and a binary file contains a testing dataset。 
- Build the NNoM with the model and do inference with the testing dataset.

After a few minutes, if everything ok, it will finally print both validation accuracy on Keras and NNoM.
~~~
Top 1 Accuracy on Keras 98.50%
Top 1 Accuracy on NNoM  98.53%
~~~

The first accuracy is the floating-point validation on Keras. The second accuracy is the fixed-point inference result from NNoM. If you see the above results, you have successfully built an NNoM model and do inference with it. (Don’t be surprised if you see fixed-point inference is giving better results..)

## How does it work?

The details can be found in “main.py”. It contains a few sections.

1. Build and train a Keras model

~~~
# The data, split between train and test sets:
(x_train, y_train), (x_test_original, y_test_original) = mnist.load_data()
...
x_test = x_test/255
x_train = x_train/255
...
model = build_model(x_test.shape[1:])
train(model, x_train,y_train, x_test, y_test, epochs=epochs)
~~~

We first load the MNIST dataset (if it is the first time, Keras will download the dataset for you), then scales them int to a range of 0 ~ 1 by dividing 255. (MNIST dataset is 28 x 28 x 8bit grey scale image). Instead of 0 ~ 255, now images are represented by 0 ~ 1. Then build the model and train it the same way as most of the other tutorials you can find online. This section will finally save the trained model in our working directory.

If you are not familiar with Keras model, you can check the “build_model()” in “main.py” for detail. This example shows a simple feed-forward model, however, complex models will be the same.

2. Generate the NNoM model

~~~
# -------- generate weights.h (NNoM model) ----------
# get the best model
model_path = os.path.join(save_dir, model_name)
model = load_model(model_path)

# evaluate
evaluate_model(model, x_test, y_test)

# save weight
generate_model(model,  x_test, format='hwc', name="weights.h")

# generate binary
generate_test_bin(x_test*127, y_test, name='mnist_test_data.bin')
~~~

Firstly, we load the best mode saved by the previous step, then evaluate the model on PC using Keras. It is optional, just for the later comparison with NNoM model.

Secondly, call the `generate_mode()` provided by NNoM to generate the NNoM model, aka the c header file `weights.h`. This is the only file contains all you need to deploy an NN model with NNoM.

Thirdly, call `generate_test_bin()` to export the dataset to a binary file which contains labels and test dataset. This file is organized as `repeated(128 bytes label + 128xdata)`.
Data format in a testing binary file

3. Build NNoM with deployed model

~~~
# --------- for test in CI ----------
# build nnom
os.system("scons")

# do inference
cmd = ".\mnist.exe" if 'win' in sys.platform else "./mnist"
if(0 == os.system(cmd)):
    result = np.genfromtxt('result.csv', delimiter=',', skip_header=1)
    result = result[:,0] # the first column is the label
    label = y_test_original
    acc = np.sum(result == label)/len(result)
    print("Top 1 Accuracy using NNoM  %.2f%%" %(acc *100))
~~~
It simply calls `scons` to compile NNoM with the model file, `weights.h`. Once the compiling completed, it runs the NNoM and uses the exported dataset by the last step to do the inference. The code is shown in `example/auto_test/main.c`. The results of each inference will be stored in a `result.csv` file. The first row is the predict labels, the second row is the probabilities of this label.

After the inference, the python code will collect the result using `result = np.genfromtxt()` then count the number of differences in the predicted labels and the labels from the original testing dataset. Finally, it prints out the inference accuracy to the screen.

The output of this example can be found in the [NNoM’s Travis CI](https://travis-ci.org/majianjia/nnom).

## Useful Logs from NNoM

> *We don’t guess whether our ad-hoc model is small enough for our MCUs. We visualise it.*

One of the advantages in NNoM it provides multiple evaluation methods to vsualise the fixed-point mode. They provide Top-K accuracy, Confusion Matrix, runtime stats and other info. For details, please check the [evaluation documentations](https://majianjia.github.io/nnom/api_evaluation/).

You can check the `example/auto_test/main.c` for a first impression.

### Can the model fit into my MCU?

During NNoM compiling, the model information will be printed on the screen. Including the output shape, mac-ops and memory utilisation.
~~~
model = nnom_model_create();
~~~

~~~
Start compiling model...
Layer(#)         Activation    output shape    ops(MAC)   mem(in, out, buf)      mem blk lifetime
-------------------------------------------------------------------------------------------------
#1   Input      -          - (  28,  28,   1)          (   784,   784,     0)    1 - - -  - - - - 
#2   Conv2D     -          - (  26,  26,  16)      97k (   784, 10816,    36)    1 1 1 -  - - - - 
#3   Conv2D     - ReLU     - (  24,  24,  32)    2.65M ( 10816, 18432,   576)    1 1 1 -  - - - - 
#4   MaxPool    -          - (  12,  12,  32)          ( 18432,  4608,     0)    1 1 1 -  - - - - 
#5   Conv2D     - ReLU     - (  10,  10,  64)    1.84M (  4608,  6400,  1152)    1 1 1 -  - - - - 
#6   Conv2D     - ReLU     - (   8,   8,  64)    2.35M (  6400,  4096,  2304)    1 1 1 -  - - - - 
#7   MaxPool    -          - (   4,   4,  64)          (  4096,  1024,     0)    1 1 1 -  - - - - 
#8   Dense      - ReLU     - (  64,   1,   1)      65k (  1024,    64,  2048)    1 1 1 -  - - - - 
#9   Dense      -          - (  10,   1,   1)      640 (    64,    10,   128)    1 1 1 -  - - - - 
#10  Softmax    -          - (  10,   1,   1)          (    10,    10,     0)    1 1 - -  - - - - 
#11  Output     -          - (  10,   1,   1)          (    10,    10,     0)    1 - - -  - - - - 
-------------------------------------------------------------------------------------------------
Memory cost by each block:
 blk_0:2304  blk_1:18432  blk_2:10816  blk_3:0  blk_4:0  blk_5:0  blk_6:0  blk_7:0  
 Total memory cost by network buffers: 31552 bytes
~~~

### How is the accuracy of the fixed-point model?

NNoM provide a set of prediction API for a complete evaluation with a whole dataset. After the inference with the whole dataset, call `prediction_summary()`, will print the Top-K accuracy and Confusion Matrix. A simplified code snip is shown below.

~~~C
// mnist, 10 classes, record top-4 accuracy
pre = prediction_create(model, nnom_output_data, 10, 4);
while(more_data){
    prediction_run(pre, true_label[i], &label, &prob);
}
prediction_end(pre);
prediction_summary(pre);
~~~

Log return by `prediction_summary()`
~~~
Prediction summary:
Test frames: 10000
Test running time: 0 sec
Model running time: 0 ms
Average prediction time: 0 us
Top 1 Accuracy: 98.53% 
Top 2 Accuracy: 98.83% 
Top 3 Accuracy: 98.97% 
Top 4 Accuracy: 99.07% 
Confusion matrix:
predict     0     1     2     3     4     5     6     7     8     9
actual
   0 |    975     0     2     0     0     0     0     2     1     0   |  99%
   1 |      0  1123     1     3     0     0     1     6     1     0   |  98%
   2 |      1     0  1024     0     0     0     0     7     0     0   |  99%
   3 |      0     0     3  1000     0     1     0     5     1     0   |  99%
   4 |      0     0     2     0   955     0     1     2     0    22   |  97%
   5 |      1     0     0     9     0   880     1     1     0     0   |  98%
   6 |     11     4     1     1     1     6   932     0     2     0   |  97%
   7 |      0     1     5     0     0     0     0  1020     0     2   |  99%
   8 |      1     0     3     2     0     1     0     4   954     9   |  97%
   9 |      3     2     0     2     2     4     0     6     0   990   |  98%
~~~

### How long does it takes for an inference?

In `main.c` we also call `model_stat()`, which returns the information below. It list the time cost to run each layer and the computational complexity of each layer in the last inference. Since we have yet port an microsecond timestamp in this example, the timing is not available in the log here.
~~~
model_stat(model);
~~~
Log from `model_stat()`
~~~
Print running stat..
Layer(#)        -   Time(us)     ops(MACs)   ops/us 
--------------------------------------------------------
#1        Input -         0                  
#2       Conv2D -         0          97k     
#3       Conv2D -         0        2.65M     
#4      MaxPool -         0                  
#5       Conv2D -         0        1.84M     
#6       Conv2D -         0        2.35M     
#7      MaxPool -         0                  
#8        Dense -         0          65k     
#9        Dense -         0          640     
#10     Softmax -         0                  
#11      Output -         0                  
Summary:
Total ops (MAC): 7020224(7.02M)
Prediction time :0us
~~~


To evaluate the runtime statistics, you have to provide a microsecond timestamp porting to “nnom_port.h”. An example with these timing info showing is in the documentation(evaluation example), which shows how does it perform with timestamp provided in an MCU .
