# Speech Noise Suppression Example

Also know as `rnn-denoise example`. [中文文档](README_CN.md)

This example is partially based on the methodology provided by the well-known [RNNoise](https://jmvalin.ca/demo/rnnoise/) project and their [paper](https://arxiv.org/abs/1709.08243) . Great thanks to them!

Audio Demo: https://youtu.be/JG0mSZ1ZnrY

**Third party packages and license**

The below third party packages used in this example are mostly released with permissive license but one who use this example must take full responsibility of following the original term and condition of these packages. 

- [RNNoise](https://jmvalin.ca/demo/rnnoise/) 
- [Microsoft Scalable Noisy Speech Dataset](https://github.com/microsoft/MS-SNSD) 
- [python speech features](https://github.com/jameslyons/python_speech_features)
- [arduino_fft](https://github.com/lloydroc/arduino_fft)
- [CMSIS](https://github.com/ARM-software/CMSIS_5)

This example and NNoM are provided under Apache-2.0 license. Please read license file in the main [NNoM repository](https://github.com/majianjia/nnom). 


# A few key points before you start

## How does noise suppression works with Neural Network?

The [RNNoise](https://jmvalin.ca/demo/rnnoise/)  has already explained the methodology that we used in this project. 

Basically, we use a neural network model to control an audio Equalizer (EQ) in a very high frequency, therefore to suppress those bands contained noise while keep the gains contain speech. 

Unlike the conventional neural networks which try output the signal directly, our NN model instead output the gains for each filter band of the equalizer.

This example uses MFCC (Mel-scale) to determine the gains, instead of Opus scale (RNNoise) or Bark scale. 

![](figures/general_workflow.png)


# Step by step Guide

## 1. Get the Noisy Speech

This example uses [Microsoft Scalable Noisy Speech Dataset](https://github.com/microsoft/MS-SNSD) (MS-SNSD). If you want to train your own model, you may download the dataset from the above repository, then put them into folder `MS-SNSD/`.

After the MS-SNSD is downloaded, you can now generate the `clean speech` and its corresponding `noisy speech` with variable levels of noise mixed.

The advantage of using MS-SNSD is it is scaleable. Modify `noisyspeech_synthesizer.cfg` to configure how does the speech generated. The recommended configurations  for the this example are:
~~~
sampling_rate: 16000
audioformat: *.wav
audio_length: 60 
silence_length: 0.0
total_hours: 15
snr_lower: 0
snr_upper: 20
total_snrlevels: 3  
~~~

Then, run `noisyspeech_synthesizer.py` to generate the speeches. If everything goes well, there will be 3 new folders created `/Noise_training`, `/CleanSpeech_training` and `NoisySpeech_training`. We only need the 2 later folders. 

## 2. Generate the dataset

Now we have Clean and Noisy speech located in `MS-SNSD/CleanSpeech_training` and `MS-SNSD/NoisySpeech_training`. They are the raw PCM signal, but our NN take MFCCs and their derivatives as input and the equalizer gains as output, so we need to process them to get our training dataset. 


Now we need to run `gen_dataset.py` to get the MFCC and gains. It will generate the dataset file `dataset.npz` which can be used later for NN training. Also,
- You can config how much MFCC features you want to use in RNN ( which is also the number of filter bands in the equlizer). Modify `num_filter = 20`; this number can be from `10` to `26`. 
- This step generate the filter coefficence of the equalizer into a file `equalizer_coeff.h` (`generate_filter_header(...)`), which will be used in C equalizer. 

In addition, `gen_dataset.py` also generates an audio file `_noisy_sample.wav` which is the raw noisy speech, as well as a filtered file filtered using the truth gains `_filtered_sample.wav`. I would recommend to play both files and see what is the best result we can get by using this equalizer suppression method. 

## 3. Training

Once we have `dataset.npz` ready, just run `main.py` to train the model. The trained model will be saved as `model.h5`

We train the RNN with `stateful=True` and `timestamps=1`, which is not friendly for training with back propagation, so I set the `batchsize` to very large to make BP's life easier. 

At the later part of `main.py`, it will reload the `model.h5` and try to process the above noisy file `_noisy_sample.wav` using our flesh trained RNN ` filtered_sig = voice_denoise(...)` and save the filtered signal into `_nn_filtered_sample.wav`. 

Also, it will use the NNoM model converter script `generate_model(...)` to generate our NNoM model `weights.h`

## 4. Inference using NNoM

This example provided a `SConstruct` so you can run `scons` in this folder to build a binary executable. 

This executable can take `.wav` file as input and output the filtered `.wav` file. 
> The **only** format it supports is **16kHz**, **1CH**. `main.c` will not parse wav file but only copy the header and jumps to the data chunk.  

Once compiled, use the below command in the folder to run
- Win powershell: `.\rnn-denoise [input_file] [output_file]` or drag the wav file onto the executable. 
- Linux: I don't know 

i.e. run `.\rnn-denoise _noisy_sample.wav _nn_fixedpoit_filtered_sample.wav`

If you have followed the guide, you show have the below files in your `rnn-denoise` folder. 
~~~
_noisy_sample.wav  --> original noisy speech
_filtered_sample.wav  --> denoised speech by truth gains
_nn_filtered_sample.wav   --> denoised speech by the NN gains from Keras
_nn_fixedpoit_filtered_sample.wav   --> denoised speech by NN gains from NNoM 
~~~

Audio example: [Bilibili](https://www.bilibili.com/video/BV1ov411C7fi), [Youtube](https://youtu.be/JG0mSZ1ZnrY)


Graphically, the results look like:

![](figures/speech_comparison.png)


# Further details

Overall, I would suggest you go through both python code, `gen_dataset.py` and `main.py`, read the comments in the code and test with different configurations. 

## Training data

`x_train` is consist of 13 or 20 MFCC coefficent and the first and second derivative of the first 10 coefficent, giving in total 33 or 40 features.  

![](figures/speech_to_dataset.png)

`y_train` is consist of 2 data, `gains` and `VAD`. 
- Gains are calculate by the sqaure root of each band energy of clean speech / noisy speech. Same as RNNoise. 
- VAD is generated by where the clean speech total energy above a threshold. It is expanded forward and backward for a few samples. 

![](figures/vad_sample.png)

## Gains and Voice Activity Detection

In the default model, there is a secondary output which has only one neural, indicating whether *speech is detected*. `1` is detected, `0` is not detected. In the MCU example `main_arm.c`, this is linked to an onboard LED which turns on if `VAD neural > 0.5`. 

![](figures/gains_vad_sample.png)

## Equalizer

The example uses a 20 bands (default) or 13 bands (or any number of band you would like) to suppress the noise. Each frequency band is filtered by 1 (default) or 2 (not stable) orders IIR bandpass filter. The -3dB (cutoff) of 2 neighbour bands are intersected. The frequency response is shown here:

![](figures/equalizer_frequency_response.png)

The signal will pass each band in parallel then added together.
Due to the overlapping of each band, the signal might sound too loud and will leed to overflow (cracking noise). So a factor of `0.6` is multipled to the final signal (not a mathematically calculated number). 

## Model structure

This example provides 2 different RNN models. The first one is RNNoise-like structure, consist of multiple links between many individual GRU layers. Those links are merge using concatenation. In addition to the gains of the equalizer, it also outputs the Voice Activity Detection(VAD). This model is the default in `main.py` The structure is shown below.  This model has around `120k` weights, larger than RNNoise because it go an extra input layer. However, the scale for the model is far beyond enough. You may reduce many GRU units without any noticeable different in result.  

![](figures/model_structure_full.png)

The other is a simple stacked-GRU. This will not provide VAD output. Surprisingly, it also work well. 
To test this one, replace the line `history = train(...)` with `train_simple(...)` in `main.py`

The Keras model is train under `stateful=True` and with `timestamp=1` which is not ideal. The `batch_size` is equal to the actual timestamp, so we need to make the timestamp as large as possible to let the BP working. We are using `batch_size > 1024`. Remember to turn off the `shuffle`. 

 > The model only take 1 epoch to be overfitted... Recurrent dropout is very much needed. You may also add regular dropout layers in between each GRU layers.

## MCU examples

The MCU example is `main_arm.c`. This example is running on STM32L476-Discovery. No RTOS related dependency.  

It uses the input from the onboard Microphone and trys to filter the signal. Currently, it only outputs a VAD through the green LED (PE8), and it **does not** record the filtered voice or playback. However, the RNN and equalizer is implemented and the signal are filtered, one can implement their own playback or recorder on the existing output  data easily.

The functional part of `main_arm.c` are identical to the `main.c`

If you are using ARM-Cortex M chips, turn on the below optimization will help to improve the performance. 

- Turn on CMSIS-NN Support for NNoM. See [Porting and Optimization Guide](../../docs/Porting_and_Optimisation_Guide.md)
- In `mfcc.h` turn on `PLATFORM_ARM` to use ARM FFT


## Performance

The performance on MCU is all we care about. No matter how good a NN model is, if our MCU cannot run it in time, it will be meaningless of all the effort we done here. 

There are 3 computational-expensive parts *MFCC*, *Neural Network*, *Equalizer(EQ)*. So I made a test to evaluate the time comsuming of these 3 parts.

The test environment:
- Board: [STM32L476-Discovery](https://www.st.com/en/evaluation-tools/32l476gdiscovery.html)
- MCU: STM32L476, overclocked @140MHz Cortex-M4F
- Audio src: Embedded Microphone
- Audio output: None (you could port it to the audio jack)
- IDE: Keil MDK


Test conditions: 
- NN backend: CMSIS-NN or Local C Backend
- FFT lib: `arm_rfft_fast_f32` or pure C fft [arduino_fft](https://github.com/lloydroc/arduino_fft)
- Tested Compiler Optimization: `-O0/-O1` or  `-O2`
- Tested equalizer bands: `13 band` or `20 band`

Remember, our input audio format is `16kHz 1CH`, which means for each audio update (`256` sample), we only have `256/16000 = 16ms` to complete the whole work.

**13 Band Equalizer**

| NN backend| 512-FFT | Opt | MFCC(ms) | Network(ms) |Equalizer(EQ)(ms)| Total(ms) | Comment|
| ------ | --- | ------ | ------ | ------| ------|  ------| ------|
|cmsis-nn|arm_fft| -O1| 0.63 | 3.34 | 2.75 | 7.11 |  |
|cmsis-nn|arm_fft |-O2| 0.56 | 3.3 | 2.27 | 6.18 |  |
|local|arm_fft|-O1| 0.63 | 7.78 | 2.75| 11.19 |  |
|local|arm_fft|-O2| 0.55 | 7.78| 2.27| 10.65|  |
|local|arduino_fft| -O0| 2.54 | 7.94 | 4.03| 14.57 ||
|local|arduino_fft| -O2| 1.89 | 7.78| 2.27| 11.98| |

 
The test results are quite impressive. With the most optimized option, the total run time is only `6.18ms`, which is around `38%` of total CPU load. Under pure C implementation  (local+arduino_fft), the total run time is still under `16ms`. 

**20 Band Equalizer**

| NN backend| 512-FFT | Opt | MFCC(ms) | Network(ms) |Equalizer(EQ)(ms)| Total(ms) | Comment|
| ------ | ------ | ------ | ------ | ------| ------|  ------| ------|
|cmsis-nn| arm_fft|-O1| 0.66 | 3.74 | 4.20 | 8.64 |  |
|cmsis-nn| arm_fft|-O2| 0.58 | 3.35 | 3.46 | 7.44 |  |
|local| arm_fft|-O1| 0.67 | 7.91 | 4.20| 12.81 |  |
|local| arm_fft|-O2| 0.59 | 7.92| 3.46| 12.02|  |
|local| arduino_fft|-O0| 2.60 | 8.09 | 6.15| 16.89 | |
|local| arduino_fft|-O2| 1.92 | 7.92| 3.47| 13.3| |


Compare the both, `20` band has more impact on the equalizer other than NN.


~~~
 \ | /
- RT -     Thread Operating System
 / | \     4.0.0 build Sep 17 2020
 2006 - 2018 Copyright by rt-thread team
RTT Control Block Detection Address is 0x20002410
Model version: 0.4.2
NNoM version 0.4.2
Data format: Channel last (HWC)
Start compiling model...
Layer(#)         Activation    output shape    ops(MAC)   mem(in, out, buf)      mem blk lifetime
-------------------------------------------------------------------------------------------------
#1   Input      -          - (   1,   1,  33,)          (    33,    33,     0)    1 - - -  - - - - 
#2   RNN/GRU    -          - (   1,  96,     )      84k (    33,    96,  1794)    1 1 3 -  - - - - 
#3   RNN/GRU    -          - (   1,  24,     )      11k (    96,    24,   624)    1 2 3 -  - - - - 
#4   RNN/GRU    -          - (   1,  24,     )     6600 (    24,    24,   480)    1 2 2 3  - - - - 
#5   Concat     -          - (   1, 144,     )          (   144,   144,     0)    1 1 2 3  - - - - 
#6   RNN/GRU    -          - (   1,  48,     )      39k (   144,    48,  1152)    1 1 1 2  1 - - - 
#7   Concat     -          - (   1, 168,     )          (   168,   168,     0)    1 - 1 2  1 - - - 
#8   RNN/GRU    -          - (   1,  64,     )      65k (   168,    64,  1488)    1 1 1 1  - - - - 
#9   Flatten    -          - (  64,          )          (    64,    64,     0)    - - 1 1  - - - - 
#10  Dense      - HrdSigd  - (  13,          )      832 (    64,    13,   128)    1 1 1 1  - - - - 
#11  Output     -          - (  13,          )          (    13,    13,     0)    - 1 - 1  - - - - 
#12  Flatten    -          - (  24,          )          (    24,    24,     0)    - 1 - 1  - - - - 
#13  Dense      - HrdSigd  - (   1,          )       24 (    24,     1,    48)    1 1 1 1  - - - - 
#14  Output     -          - (   1,          )          (     1,     1,     0)    - 1 1 -  - - - - 
-------------------------------------------------------------------------------------------------
Memory cost by each block:
 blk_0:624  blk_1:1796  blk_2:96  blk_3:24  blk_4:48  blk_5:0  blk_6:0  blk_7:0  
 Total memory cost by network buffers: 2588 bytes
Compling done in 176 ms
~~~

~~~
Print running stat..
Layer(#)        -   Time(us)     ops(MACs)   ops/us 
--------------------------------------------------------
#1  Input      -         2                  
#2  RNN/GRU    -      1034          84k     81.59
#3  RNN/GRU    -       251          11k     46.94
#4  RNN/GRU    -       149         6600     44.59
#5  Concat     -         4                  
#6  RNN/GRU    -       706          39k     56.22
#7  Concat     -         4                  
#8  RNN/GRU    -      1099          65k     59.80
#9  Flatten    -         1                  
#10 Dense      -        26          832     32.00
#11 Output     -         1                  
#12 Flatten    -         0                  
#13 Dense      -         4           24     6.00
#14 Output     -         1                  

Summary:
Total ops (MAC): 208952(0.20M)
Prediction time :3281us
Efficiency 63.68 ops/us
Total memory:6512
Total Memory cost (Network and NNoM): 6512

~~~













