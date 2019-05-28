## Key-Word Spotting (KWS) example

> This example is under development, the codes are supposed to be changed during the period.

This example will show and example to use Neural Network to recognise Speech Commands, or so called Key-Word Spotting (KWS).

This **is not** an example for Speech to Text or Natural Language Processing. 

The target application of KWS are: low-power wakeup, speech control toys, 


## How does neural network recognise speech commands?

- It use microphones on development board to capture continuous voice. 
- Then the voice will be splited and then converted to Mel-frequency cepstral coefficients([MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)).
- The output of MFCC are similar to a image.
- Use neural network recognise these MFCC same as to classify images. 

A one second MFCC result is shown below:
![](kws_mfcc_example1.png)
![](kws_mfcc_example2.png)
![](kws_mfcc_example3.png)

The size of image is `(63, 12, 1)`. 
So there is no difference in the Neural Network in terms of image classification and speech commands recognition. What's more in KWS are the mic driver and MFCC. 

## Preperations

Download the [Google speech command dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz).
create a folder `dat` then unzip the above dataset into the folder. 

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

The dataset contains many voice commands and noises records. Most of the records are length 1 seconds or less. 

The available labels are:
~~~
'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight','five', 'follow', 'forward',
'four','go','happy','house','learn','left','marvin','nine','no','off','on','one','right',
'seven','sheila','six','stop','three','tree','two','up','visual','yes','zero'
~~~

## Microphone

The Audio format in the example is `16bit, 16kHz, 1CH`


## MFCC

I would suggest you google/baidu to have further understanding of the MFCC. 

The short explaination is it takes voice input (1D, time-sequence data) generate a image (2D) using mel[ody] scale. What make it different from simple FFT is it also scales the frequency in order to match more closely what the human ear can hear.

In this example, the windows size is `31.25ms` which is `512 point at 16kHz`. The windows over lapping is `50%` or `16.125ms`. Filter band number `26`, filter band low frequency `20Hz`, filter band high freqency `4000`. 

You should be familiar with the above parameters if you have checked any MFCC tutorials. If you haven't, it doesn't matter, just leave them as they were. 

> MFCC must be implemented in both MCU and PC. 

As said before, the datasets are all equal to or below 1 sec. Before training, these voice commands are padding to 1 seconds which have all len = `16000`. If we split the 16000 into size of 512 with overlapping of 50%, we will get `63` pieces.  

Normally, we takes 13 coefficients in each windows `512 point or 31.25ms`(see how much the data is compressed !). But we will ignore the first one. I dont know why but peoples are doing this (maybe the very low frequency are less meaningful in speech?). So, every `512` audio data we got `12` coefficients.

Then it is easy to guess why the MFCC "image" has `63` in width (timestep) and `12` in height (coefficients). 



 




















