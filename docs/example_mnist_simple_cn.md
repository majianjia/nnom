
# MNIST-SIMPLE

这是一个最简单的例子。

## 1. 下载并启用NNoM

在 RT-Thread 的包管理中

~~~
RT-Thread online packages  --->
    miscellaneous packages  --->
        [*] NNoM: A Higher-level Nerual Network ...	--->

*选择 latest 版本
*需要打开 msh 支持		
~~~

源码请到[github](https://github.com/majianjia/nnom)


## 2. 复制三个工程文件

把 `packages/nnom-latest/examples/mnist-simple/mcu` 目录下的 `image.h`, `weight.h`和 `main.c` 复制到工程目录的 `application/`。替换掉默认的 `main.c`。先不用管内容。

（如果你是好奇宝宝：）

> `image.h` 里面放置了 10 张从 MNIST 数据集里面随机挑选的图片。
>
> `weight.h` 是 NNoM 的工具脚本自动生成的模型参数。
> 
> `main.c` 包含了最简单的模型初始化和 msh 交互命令。


## 3. 跑起来

编译，下载，运行

RT-Thread 启动后， 会接着打印包含在 `weight.h` 里面的模型的编译信息。

### 3.1 模型编译

在 `main()`函数里面，调用了 `model = nnom_model_create();`。这句话将会载入我们藏在 `weight.h` 里面的模型，将它编译并把信息打印出来。

~~~
 \ | /
- RT -     Thread Operating System
 / | \     4.0.0 build Mar 29 2019
 2006 - 2018 Copyright by rt-thread team
RTT Control Block Detection Address is 0x20000a8c
msh >
INFO: Start compile...
Layer        Activation    output shape      ops          memory            mem life-time
----------------------------------------------------------------------------------------------
 Input      -          - (  28,  28,   1)        0   (  784,  784,    0)    1 - - -  - - - - 
 Conv2D     - ReLU     - (  28,  28,  12)    84672   (  784, 9408,  432)    1 1 - -  - - - - 
 MaxPool    -          - (  14,  14,  12)        0   ( 9408, 2352,    0)    1 - 1 -  - - - - 
 Conv2D     - ReLU     - (  14,  14,  24)   508032   ( 2352, 4704,  864)    1 1 - -  - - - - 
 MaxPool    -          - (   7,   7,  24)        0   ( 4704, 1176,    0)    1 - 1 -  - - - - 
 Conv2D     - ReLU     - (   7,   7,  48)   508032   ( 1176, 2352, 1728)    1 1 - -  - - - - 
 MaxPool    -          - (   4,   4,  48)        0   ( 2352,  768,    0)    1 - 1 -  - - - - 
 Dense      - ReLU     - (  96,   1,   1)    73728   (  768,   96,  768)    1 1 - -  - - - - 
 Dense      -          - (  10,   1,   1)      960   (   96,   10,   96)    1 - 1 -  - - - - 
 Softmax    -          - (  10,   1,   1)        0   (   10,   10,    0)    - 1 - -  - - - - 
 Output     -          - (  10,   1,   1)        0   (   10,   10,    0)    1 - - -  - - - - 
----------------------------------------------------------------------------------------------
INFO: memory analysis result
 Block0: 1728  Block1: 2352  Block2: 9408  Block3: 0  Block4: 0  Block5: 0  Block6: 0  Block7: 0  
 Total memory cost by network buffers: 13488 bytes
~~~

这里面的信息有：

- 模型有三个卷积层组成，每个卷积层都使用 ReLU 进行激活 （ReLU： 大于0的数值不变，小于0的数值重新赋值为0）。
- 三个卷积后面跟着两个 Dense 层 （Densed-connected， 也叫 fully-connected 全连接层）。
- 最后模型通过 Softmax 层来输出 （将数值转换成概率值）
- 各层的内存信息，计算量 （定点乘加操作：MAC-OPS）
- 总网络内存占用 13488 bytes

### 3.2 跑个模型

之前我们介绍过 `image.h` 里面有十张图片。我们现在可以通过 `mnist` 这个命令来用他们测试一下上面的模型。

命令如下, num 是 0~9 里面的任意数字。代表第几个图片（注意，并指非数字几，图片是随机拉取的）

`mnist num`

我输入了 

`msh >mnist 6`

~~~
msh >mnist 6

prediction start.. 
                                                        
                                                        
                                                        
                                                        
                                                        
                                ..]]  ((ZZOO))^^        
                          ``//qq&&))  kkBB@@@@ff        
                    "">>\\pp%%ZZ,,    [[%%@@BB11        
                ^^}}MM@@@@oo{{      rr@@@@OO<<          
                nn@@@@aajj..    ++dd@@88nn''            
              \\%%@@hh!!      ++88@@oo::                
            !!%%@@kk>>      ;;88@@oo::                  
            ))@@@@<<      ^^pp@@oo::                    
            ::oo@@WWzzll!!bb@@bb''                      
              ttBB@@@@%%WW@@**,,                        
                ll}}LL%%@@@@@@bbtt''                    
                    ``&&@@MMCC&&%%hh[[                  
                    ((@@@@((    II**@@nn''              
                    ??@@##``        QQ@@>>              
                    ((@@@@^^        [[@@pp              
                    [[@@@@^^        nn@@jj              
                    ..aa@@[[        ZZ%%++              
                      __@@**,,    xx@@OO                
                        {{&&**jj00@@aa::                
                          ^^YYpppp||,,                  
                                                        
                                                        
                                                        
Time: 62 tick
Truth label: 8
Predicted label: 8
~~~

额，如果恶心到你了我道歉...

输出的信息里面记录了

- 此次预测的时间，这里用了 `62 tick`，我这是相当于 62ms
- 这张图片的真实数字是 `8`
- 网络计算的这张照片的数字 `8`

赶快去试试，其他的 9 张图片吧。


简单的体验就到这。

## 4 建立自己的模型

想要在单片机上跑自己的模型，需要先学会在 Keras 里面建立一个自己的模型。

参考 `nnom/example/mnist-simple/model` 里面的 `mnist_simple.py` 自行修改。

*记得把 `nnom/scripts` 下的几个文件复制到以上目录。

训练完成后，会生成 `weights.h` 还会生成随机图片文件 `image.h`。 接下来按照上面的操作从头来一遍就好。

## 5 其他

更多例子，高级用法，请查看[文档](https://majianjia.github.io/nnom/)和[其他例子](https://github.com/majianjia/nnom/tree/master/examples)




















