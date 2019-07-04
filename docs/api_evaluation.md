
## Evaluation tools

NNoM has provide a few evaluation interfaces. Thye can either do runtime statistic or model evaluations. 

These API are print though the standard `printf()`, thus a terminal/console is needed. 

All these API must not be called before the model has been compiled. 

---

## model_stat()
~~~C
void model_stat(nnom_model_t *m);
~~~

To print the runtime statistic of the last run. Check the below example for the availble statistics. 

**Arguments**

- ** m:** the model to print.


**Notes**

It is recommended to run the mode once after compiling to gain these runtime statistic. 


**Example**
~~~
Print running stat..
Layer(#)        -   Time(us)      ops(MACs)     ops/us 
--------------------------------------------------------
#1        Input -         9              0      
#2       Conv2D -      8292         172800      20.83
#3      MaxPool -      5753              0      
#4       Conv2D -     50095        3612672      72.11
#5      MaxPool -      3617              0      
#6       Conv2D -     35893        2654208      73.94
#7      MaxPool -      1206              0      
#8       Conv2D -      4412         171072      38.77
#9   GL_SumPool -        30              0      
#10     Softmax -         2              0      
#11      Output -         0              0      

Summary.
Total ops (MAC): 6610752
Prediction time :109309us
Efficiency 60.47 ops/us
Total Memory cost (Network and NNoM): 32876
~~~

---

## nnom_predict()

~~~C
int32_t nnom_predict(nnom_model_t *m, uint32_t *label, float *prob);
~~~

A standalone evaluation method, run single prodiction, return probability and top-1 label. 

This method is basicly `model_run()` + `index(top-1)`

**Arguments**

- **m:** the model to run prediction (evaluation).
- **label:** the variable to store top-1 label.
- **prob:** the variable to store probability. Range from 0~1.

**Return**

- The predicted label in digit. 
- Error codes if model is failed to run.

**Note**

The input buffer of the model must be feeded before calling this method. 

---

## prediction_create()

~~~C
nnom_predict_t *prediction_create(nnom_model_t *m, int8_t *buf_prediction, 
		size_t label_num, size_t top_k_size);
~~~

This method create a prediction instance, which record mutiple parameters in the evaluation process. 


**Arguments**

- **m:** the model to run prediction (evaluation).
- **buf_prediction:** the output buffer of the model, which should be the output of Softmax. Size equal to the size of class. 
- **label_num:** the number of labels (the number of classifications).
- **top_k_size:** the Top-k that wants to evaluate.

**Return**

- The prediction instance.

**Note**

Check later examples. 

---

## prediction_run()

~~~c
nnom_status_t prediction_run(nnom_predict_t *pre, 
	uint32_t true_label, uint32_t* predict_label, float* prob)
~~~

To run a prediction with the new data (feeded by user to the input_buffer which passed to Input layer).

**Arguments**

- **pre:** the prediction instance created by `prediction_create()`.
- **true_label:** true label of this data.
- **predict_label:** the predicted label of this data (top-1 results).
- **prob:** the probability of this label.

**Return**

- nnom_status_t 

---

## prediction_end()

~~~c
void prediction_end(nnom_predict_t *pre);
~~~

To mark the prediction has done. 


**Arguments**

- **pre:** the prediction instance created by `prediction_create()`.

---


## prediction_delete()

~~~c
void predicetion_delete(nnom_predict_t *pre);
~~~

To free all resources. 

**Arguments**

- **pre:** the prediction instance created by `prediction_create()`.

---


## prediction_matrix()

~~~C
void prediction_matrix(nnom_predict_t *pre);
~~~

To print a confusion matrix when the prediction is done. 

**Arguments**

- **pre:** the prediction instance created by `prediction_create()`.

**Example**

~~~
Confusion matrix:
predic     0     1     2     3     4     5     6     7     8     9    10
actual
   0 |   395     1     0     0     2     0     0     0     0     0    21   |  94%
   1 |     0   355     4     7     1     0     0     0     0     3    35   |  87%
   2 |     0     3   325     2     1     0     7    29     3     2    53   |  76%
   3 |     0    33     1   335     1     0     0     0     0     2    34   |  82%
   4 |     6     0     1     0   371     3     0     0     0     0    31   |  90%
   5 |     0     0     2     0     6   347     0     0     0     0    41   |  87%
   6 |     0     1     5     8     0     0   322     4     0     0    56   |  81%
   7 |     0     3    23     0     3     0     9   330     1     1    32   |  82%
   8 |     0     0     5     2     0     0     0     0   343     4    57   |  83%
   9 |     0    40     4    10     1     0     0     0     0   304    43   |  75%
  10 |     4    61    16    34    28    17    14     6    12    37  6702   |  96%

~~~

---

## prediction_top_k()

~~~C
void prediction_top_k(nnom_predict_t *pre);
~~~

To print a Top-k when the prediction is done. 

**Arguments**

- **pre:** the prediction instance created by `prediction_create()`.

**Example**

~~~
Top 1 Accuracy: 92.03% 
Top 2 Accuracy: 96.39% 
Top 3 Accuracy: 97.38% 
Top 4 Accuracy: 97.85% 
Top 5 Accuracy: 98.13% 
Top 6 Accuracy: 98.40% 
Top 7 Accuracy: 98.59% 
Top 8 Accuracy: 98.88% 
Top 9 Accuracy: 99.14% 
Top 10 Accuracy: 99.60% 
~~~

---

## prediction_summary()

~~~C
void prediction_summary(nnom_predict_t *pre);
~~~

To print a summary when the prediction is done. 

**Arguments**

- **pre:** the prediction instance created by `prediction_create()`.

**Example**

~~~
Prediction summary:
Test frames: 11005
Test running time: 1598 sec
Model running time: 1364908 ms
Average prediction time: 124026 us
Average effeciency: 53.30 ops/us
Average frame rate: 8.0 Hz
~~~


---



## Example

**Evaluate a model using `prediction_*` APIs**

The model needs to be compiled before it is being evaluated. 

The evaluation gose through a few steps

1. Create a instance using `prediction_create()`
2. Feed the data one by one to the input buffer, then call `prediction_run()` with the true label. 
3. When all data has predicted, call `prediction_end()`

Then you can use `prediction_matrix()`, `prediction_top_k()`, and `prediction_summary()` to see the results.

In addition, you can call `model_stat()` to see the performance of the last prediction. 

After all, call `prediction_delete()` to release all memory. 


Log from [Key-word Spotting Example](https://github.com/majianjia/nnom/tree/master/examples/keyword_spotting). 

~~~

msh >                                                                                                                                   
 \ | /
- RT -     Thread Operating System
 / | \     4.0.0 build Mar 28 2019
 2006 - 2018 Copyright by rt-thread team
RTT Control Block Detection Address is 0x20000b3c
msh >
INFO: Start compile...
Layer        Activation    output shape      ops          memory            mem life-time
----------------------------------------------------------------------------------------------
 Input      -          - (  62,  12,   1)        0   (  744,  744,    0)    1 - - -  - - - - 
 Conv2D     - ReLU     - (  60,  10,  32)   172800   (  744,19200, 1152)    1 1 - -  - - - - 
 MaxPool    -          - (  30,   9,  32)        0   (19200, 8640,    0)    1 - 1 -  - - - - 
 Conv2D     - ReLU     - (  28,   7,  64)  3612672   ( 8640,12544, 2304)    1 1 - -  - - - - 
 MaxPool    -          - (  14,   6,  64)        0   (12544, 5376,    0)    1 - 1 -  - - - - 
 Conv2D     - ReLU     - (  12,   4,  96)  2654208   ( 5376, 4608, 3456)    1 1 - -  - - - - 
 MaxPool    -          - (   6,   3,  96)        0   ( 4608, 1728,    0)    1 - 1 -  - - - - 
 Conv2D     -          - (   6,   3,  11)   171072   ( 1728,  198,  396)    1 1 - -  - - - - 
 GL_SumPool -          - (   1,   1,  11)        0   (  198,   11,   44)    1 - 1 -  - - - - 
 Softmax    -          - (   1,   1,  11)        0   (   11,   11,    0)    - 1 - -  - - - - 
 Output     -          - (  11,   1,   1)        0   (   11,   11,    0)    1 - - -  - - - - 
----------------------------------------------------------------------------------------------
INFO: memory analysis result
 Block0: 3456  Block1: 8640  Block2: 19200  Block3: 0  Block4: 0  Block5: 0  Block6: 0  Block7: 0  
 Total memory cost by network buffers: 31296 bytes

msh >nn
nn_stat
msh >nn_stat

Print running stat..
Layer(#)        -   Time(us)      ops(MACs)     ops/us 
--------------------------------------------------------
#1        Input -         9              0      
#2       Conv2D -      8294         172800      20.83
#3      MaxPool -      5750              0      
#4       Conv2D -     50089        3612672      72.12
#5      MaxPool -      3619              0      
#6       Conv2D -     35890        2654208      73.95
#7      MaxPool -      1204              0      
#8       Conv2D -      4411         171072      38.78
#9   GL_SumPool -        30              0      
#10     Softmax -         3              0      
#11      Output -         0              0      

Summary.
Total ops (MAC): 6610752
Prediction time :109299us
Efficiency 60.48 ops/us
Total Memory cost (Network and NNoM): 32876
msh >pre
predict
msh >predict
Please select the NNoM binary test file and use Ymodem-128/1024  to send.
CCCC 
Prediction done.

Prediction summary:
Test frames: 11005
Test running time: 1598 sec
Model running time: 1364908 ms
Average prediction time: 124026 us
Average effeciency: 53.30 ops/us
Average frame rate: 8.0 Hz
Top 1 Accuracy: 92.03% 
Top 2 Accuracy: 96.39% 
Top 3 Accuracy: 97.38% 
Top 4 Accuracy: 97.85% 
Top 5 Accuracy: 98.13% 
Top 6 Accuracy: 98.40% 
Top 7 Accuracy: 98.59% 
Top 8 Accuracy: 98.88% 
Top 9 Accuracy: 99.14% 
Top 10 Accuracy: 99.60% 

Confusion matrix:
predict    0     1     2     3     4     5     6     7     8     9    10
actual
   0 |   395     1     0     0     2     0     0     0     0     0    21   |  94%
   1 |     0   355     4     7     1     0     0     0     0     3    35   |  87%
   2 |     0     3   325     2     1     0     7    29     3     2    53   |  76%
   3 |     0    33     1   335     1     0     0     0     0     2    34   |  82%
   4 |     6     0     1     0   371     3     0     0     0     0    31   |  90%
   5 |     0     0     2     0     6   347     0     0     0     0    41   |  87%
   6 |     0     1     5     8     0     0   322     4     0     0    56   |  81%
   7 |     0     3    23     0     3     0     9   330     1     1    32   |  82%
   8 |     0     0     5     2     0     0     0     0   343     4    57   |  83%
   9 |     0    40     4    10     1     0     0     0     0   304    43   |  75%
  10 |     4    61    16    34    28    17    14     6    12    37  6702   |  96%

msh >                                                                                                                                   
OO: command not found.
msh >
msh >nn
nn_stat
msh >nn_stat

Print running stat..
Layer(#)        -   Time(us)      ops(MACs)     ops/us 
--------------------------------------------------------
#1        Input -         9              0      
#2       Conv2D -      8292         172800      20.83
#3      MaxPool -      5753              0      
#4       Conv2D -     50095        3612672      72.11
#5      MaxPool -      3617              0      
#6       Conv2D -     35893        2654208      73.94
#7      MaxPool -      1206              0      
#8       Conv2D -      4412         171072      38.77
#9   GL_SumPool -        30              0      
#10     Softmax -         2              0      
#11      Output -         0              0      

Summary.
Total ops (MAC): 6610752
Prediction time :109309us
Efficiency 60.47 ops/us
Total Memory cost (Network and NNoM): 32876
msh > 


~~~



































