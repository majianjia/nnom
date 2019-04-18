# Octave Convolution example

**(Source code and models are under developing)**

![](octave_conv.png)

## Octave Conv

[Drop an Octave: Reducing Spatial Redundancy inConvolutional Neural Networks with Octave Convolution](https://arxiv.org/pdf/1904.05049.pdf)

Octave Conv is a simple and effective convolution release recently. 

Octave Conv requires the below operations:

- Convlution, or Depthwise Convlution
- Pooling
- Addition
- UpSampling
- Concatenate

NNoM has already supported the these operations without any modification. 



This is a simple octave convlution 
~~~python
def octave_conv2d(xh, xl, ch=12):
    # YH=f(XH;WH→H) +upsample(f(XL;WL→H),2)
    # YL=f(XL;WL→L) +f(pool(XH,2);WH→L))

    # f(XH;WH→H)
    xhh = Conv2D(ch, kernel_size=(3, 3), strides=(1, 1), padding='same')(xh)

    # f(XH;WH→H)
    xll = Conv2D(ch, kernel_size=(3, 3), strides=(1, 1), padding='same')(xl)

    # upsample(f(XL;WL→H),2)
    xlh = UpSampling2D(size=(2, 2))(xl)

    # f(pool(XH,2);WH→L))
    xhl = MaxPool2D(pool_size=(2, 2), padding='same')(xh)
    #xhl = AvgPool2D(pool_size=(2, 2), padding='same')(xh)

    # yh = xhh + xlh
    # yl = xll + xhl
    yh = add([xhh, xlh])
    yl = add([xll, xhl])

    return yh, yl
~~~


