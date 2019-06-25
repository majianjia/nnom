
# Porting

Porting is not necessary since NNoM is a pure C framework. 
If your development environment support standard C library (libc), it will run without any problem by default setting.

However, porting can gain better performance in some platforms by switch the backends or to provide print-out model info and evaluation.

## Options
Porting is simply done by modified the `nnom_port.h` under `port/`

The default setting is shown below. 
~~~C
// memory interfaces
#define nnom_malloc(n)      malloc(n) 
#define nnom_free(p)        free(p)
#define nnom_memset(p,v,s)  memset(p,v,s)

// runtime & debuges
#define nnom_us_get()       0
#define nnom_ms_get()       0
#define NNOM_LOG(...)       printf(__VA_ARGS__)

// NNoM configuration
#define NNOM_BLOCK_NUM      (8)		// maximum number of memory block  
#define DENSE_WEIGHT_OPT    (1)		// if used fully connected layer optimized weights. 

// Formate configuration
//#define NNOM_USING_CHW  			// using CHW format instead of default HWC
//#define NNOM_USING_CMSIS_NN       // uncomment if use CMSIS-NN for optimation 
~~~

### Memory interfaces
Memory interfaces are required.

If your platform doesn't support std interface, please modify them according to your platform. See [example](#examples) for porting for RT-Thread.

### Runtime & debuges 
They are optional. 

Its recommented to port them if your platform has console or terminal.
They will help you to validate your model. 

`nnom_us_get()` is used in runtime analysis. 
If presented, NNoM would be able to record the time cost for each layer and calcualte the effeciency of them. 
This method should return the current time in a us resolution (16/32-bit unsigned value, values can overflow). 

`nnom_ms_get()` is used in evaluation with APIs in 'nnom_utils.c'. 
It is used to evaluate the performance with the whole banch of testing data. 

`LOG()` is used to print model compiling info and evaluation info. 

### NNoM configuration
`NNOM_BLOCK_NUM` is the maximum number of memory block. The utilisation of memory block will be printed during compiling. 
Adjust it when needed. 

`DENSE_WEIGHT_OPT`, reorder weights for dense will gain better performance. If your model is using 'nnom_utils.py' to deploy, weights are already reordered. 


### Format configuration

There are 2 formats normally use in machine learning. `HWC` and `CHW`. 
HWC called channel last, CHW called channel first. 

The default backend will be running on `HWC format`, which is optimized for CPU case. Images are normally stored in the memory using HWC format. 

**1. HWC format**

`NNOM_USING_CMSIS_NN` : uncomment it will enable the CMSIS-NN backend for acceleration.

On ARM-Cortex-M4/7/33/35P chips, the performance can be increased about 5x while you enable it. 
For detail please check the [paper](https://arxiv.org/pdf/1801.06601.pdf)

To switch the backend from local backend to the optimized CMSIS-NN/DSP, simply uncomment the line `#define NNOM_USING_CMSIS_NN`.

Then, in your project, you must:

1. Include the CMSIS-NN as well as CMSIS-DSP in your project. 
2. Make sure the optimisation is enable on CMSIS-NN/DSP. 

**Notes**

It is required that CMSIS version above 5.5.1+ (NN version > 1.1.0, DSP version 1.6.0). 

Make sure your compiler is using the new version of "arm_math.h". 
There might be a few duplicated in a project, such as the STM32 HAL has its own version of "arm_math.h" 

You might also define your chip core and enable your FPU support in your pre-compile configuration if you are not able to compile. 
> *e.g. when using STM32L476, you might add the two macro in your project ' ARM_MATH_CM4,  __FPU_PRESENT=1'*


After all, you can try to evaluate the performance using the APIs in 'nnom_utils.c'

**2. CHW format**

`NNOM_USING_CHW` : uncomment it will change the whole backend format to `CHW`

This format runs very inefficient convolution in CPU only mode. However, it is compatible with most hardware accelerations, such as KPU in [K210](https://kendryte.com/). 

The pure C implemenmtation is completed. The haredware acceleration using KPU is underdevelopment, will be available soon.  

**Notes**

When enable CHW model, CMSIS-NN will be automaticly excluded. 


## Examples

### Porting for RT-Thread

~~~C
// memory interfaces
#define nnom_malloc(n)      rt_malloc(n) 
#define nnom_free(p)        rt_free(p)
#define nnom_memset(p,v,s)  rt_memset(p,v,s)

// runtime & debuges
#define nnom_us_get()       0
#define nnom_ms_get()       rt_tick_get()	// when tick is set to 1000
#define NNOM_LOG(...)       rt_kprintf(__VA_ARGS__)

// NNoM configuration
#define NNOM_BLOCK_NUM      (8)		// maximum number of memory block  
#define DENSE_WEIGHT_OPT    (1)		// if used fully connected layer optimized weights. 

// Formate configuration
//#define NNOM_USING_CHW  			// using CHW format instead of default HWC
//#define NNOM_USING_CMSIS_NN       // uncomment if use CMSIS-NN for optimation 
~~~
































