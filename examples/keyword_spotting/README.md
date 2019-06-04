## Key-Word Spotting (KWS) example

> This example is under development, the codes are supposed to be changed during the period.

Questions and discussion welcome using [Issues](https://github.com/majianjia/nnom/issues)

This example is to demonstrate using Neural Network to recognise Speech Commands, or so called Key-Word Spotting (KWS).

This **is not** an example for Speech to Text or Natural Language Processing (NLP)

The target application of KWS could be: low-power wakeup, speech control toys ...

[Test video](https://youtu.be/d9zxbZM_4D0). 


## How does this example work?

- It uses a microphone on the development board to capture continuous voice. 
- The voice will be split and then converted to Mel-Frequency Cepstral Coefficients([MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)).
- The output of MFCC is similar to an image.
- Use the neural network to recognise these MFCCs same as to classify images. 

A few one second MFCC result are shown below:

![](kws_mfcc_example1.png)
![](kws_mfcc_example2.png)
![](kws_mfcc_example3.png)

The size of the image is `(63, 12, 1)`. 
There is no difference for the Neural Network to do image classification or speech commands recognition. What's more in KWS are the mic driver and MFCC. 

## Preparations

Download the [Google speech command dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz).
Create a folder `dat` then unzip the dataset into the folder. 
The structure will similar to this:

~~~
    kws
     |- dat
     |   |- _background_noise_
     |   |- backward
     |   |- ...
     |   |- README.MD
     |   |- ...
     |- mfcc.py
     |- kws.py
~~~

The dataset contains many voice commands and noises records. Most of the records are in 1 second length or less. 

The available labels in the dataset are:
~~~
'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
'four','go','happy','house','learn','left','marvin','nine','no','off','on','one','right',
'seven','sheila','six','stop','three','tree','two','up','visual','yes','zero'
~~~

## Microphone

The Audio format in the example is `16bit, 16kHz, 1CH`

## MFCC

I would suggest to google/baidu for more info of the MFCC. 

The short explanation is it takes voice input (1D, time-sequence data) and generate an image (2D) using mel[ody] scale. What makes it different from simple FFT is it also scales the frequency in order to match more closely what the human ear can hear.

In this example, the size of the windows is `31.25ms` which is `512 point at 16kHz`. The windows's overlapping is `50%` or the moving step is `16.125ms`. Filter band number `26`, filter band low frequency `20Hz`, filter band high frequency  `4000`. 

You should be familiar with the above parameters if you have checked any MFCC tutorials. If you haven't, it doesn't matter, just leave them as they were. 

> MFCC must be implemented in both MCU and PC. 

As said before, the datasets are all equal to or below 1 sec. Before training, these voice commands are padding to 1 second which have all len = `16000`. If we split the 16000 into size of 512 with overlapping of 50%, we will get `63` pieces.  

Normally, we take 13 coefficients in each windows `512 point or 31.25ms`(see how much the data is compressed !). But we will ignore the first one. I dont know why but people are doing this (maybe the very low frequency are less meaningful in speech?). So, every `512` audio data we got `12` coefficients.

Then it is easy to guess why the MFCC "image" has `63` in width (timestamp) and `12` in height (coefficients). 


## Porting

The example under `mcu` was originally built for STM32 with DFSDM and digital microphone. The board I use was [STM32L476-Discovery](https://www.st.com/en/evaluation-tools/32l476gdiscovery.html) together with RT-Thread. 

However, it is quite straight forward to port to your development board. 
MCUs based on cortex-M only need to provide an audio source from the mic (also include CMSIS-DSP for FFT)


## Performances

**Model Performance**

Top 1 Accuracy :0.83371195

Top 2 Accuracy:0.91076785

It takes `15,020` bytes of RAM and `62,787` 8-bit weighs (ROM). 

The MAC operation is `2M`, which is equal to `4M` FLOP. 

It takes `48ms` on the STM32L467 @ 150MHz to predict 1 second of voice. 

The STM32L4 can do average `43.5 MAC operation` in one microsecond, equal to `0.29 MAC/Hz`. 

Please check the log for details.


~~~
Start compiling model...
Layer(#)         Activation    output shape    ops(MAC)   mem(in, out, buf)      mem blk lifetime
-------------------------------------------------------------------------------------------------
#1   Input      -          - (  63,  12,   1)          (   756,   756,     0)    1 - - -  - - - - 
#2   Conv2D     - ReLU     - (  59,   8,  16)     188k (   756,  7552,   100)    1 1 1 -  - - - - 
#3   MaxPool    -          - (  29,   8,  16)          (  7552,  3712,     0)    1 1 1 -  - - - - 
#4   Conv2D     - ReLU     - (  27,   6,  32)     746k (  3712,  5184,   576)    1 1 1 -  - - - - 
#5   MaxPool    -          - (  13,   6,  32)          (  5184,  2496,     0)    1 1 1 -  - - - - 
#6   Conv2D     - ReLU     - (  11,   4,  64)     811k (  2496,  2816,  1152)    1 1 1 -  - - - - 
#7   Conv2D     - ReLU     - (   9,   2,  32)     331k (  2816,   576,  2304)    1 1 1 -  - - - - 
#8   Dense      -          - (  35,   1,   1)      20k (   576,    35,  1152)    1 1 1 -  - - - - 
#9   Softmax    -          - (  35,   1,   1)          (    35,    35,     0)    1 - 1 -  - - - - 
#10  Output     -          - (  35,   1,   1)          (    35,    35,     0)    1 - - -  - - - - 
-------------------------------------------------------------------------------------------------
Memory cost by each block:
 blk_0:2304  blk_1:3712  blk_2:7552  blk_3:0  blk_4:0  blk_5:0  blk_6:0  blk_7:0  
 Total memory cost by network buffers: 13568 bytes
Compling done in 37 ms
msh >

~~~

~~~
msh >nn_stat

Print running stat..
Layer(#)        -   Time(us)     ops(MACs)   ops/us 
--------------------------------------------------------
#1        Input -        10                  
#2       Conv2D -     10194         188k     18.52
#3      MaxPool -      1967                  
#4       Conv2D -     14713         746k     50.73
#5      MaxPool -      1338                  
#6       Conv2D -     13864         811k     58.49
#7       Conv2D -      5658         331k     58.63
#8        Dense -       446          20k     45.20
#9      Softmax -        10                  
#10      Output -         2                  

Summary:
Total ops (MAC): 2098240(2.09M)
Prediction time :48202us
Efficiency 43.53 ops/us
Total Memory cost (Network and NNoM): 15020
msh > 
~~~






