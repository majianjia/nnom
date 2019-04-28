# Octave Convolution example

**(Source code and model are under developing, which could be changed from time to time)**

![](octave_conv.png)

## Octave Convolution

[Drop an Octave: Reducing Spatial Redundancy inConvolutional Neural Networks with Octave Convolution](https://arxiv.org/pdf/1904.05049.pdf)

Octave Convolution is a simple and effective convolution just released recently. 
The basic idea is to have different frequency responses on image, therefore each convolution will return high frequency output and low frequency output. 

Octave Convolution requires the below operations:

- Convlution, or Depthwise Convlution
- Pooling
- Addition
- UpSampling
- Concatenate

NNoM has already supported the these operations without any modification. 


Below is a simple octave convlution example with Keras 
~~~python
def octave_conv2d(xh, xl, ch=12):
	# one octave convolution is consist of the 2 equations
    # YH=f(XH;WH→H)+upsample(f(XL;WL→H),2)
    # YL=f(XL;WL→L)+f(pool(XH,2);WH→L))

    # f(XL;WL→L)
    xhh = Conv2D(ch, kernel_size=(3, 3), strides=(1, 1), padding='same')(xh)

    # f(XH;WH→H)
    xll = Conv2D(ch, kernel_size=(3, 3), strides=(1, 1), padding='same')(xl)

    # upsample(f(XL;WL→H),2)
	xlh = Conv2D(ch, kernel_size=(3, 3), strides=(1, 1), padding='same')(xl)
    xlh = UpSampling2D(size=(2, 2))(xlh)

    # f(pool(XH,2);WH→L))
	xhl = Conv2D(ch, kernel_size=(3, 3), strides=(1, 1), padding='same')(xh)
    xhl = AvgPool2D(pool_size=(2, 2), padding='same')(xhl)

    # yh = xhh + xlh
    # yl = xll + xhl
    yh = add([xhh, xlh])
    yl = add([xll, xhl])

    return yh, yl
~~~


