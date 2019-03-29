
## Activation Layers

To reduce the memory footprint, activations provides both **Layer APIs** and **Activation APIs**. 

**Layer APIs** will create a layer instance for running the activation; 
**Activation APIs** will only create activation instance, which can be attached on a existing layers. 


---

# Activations' Layer APIs

---

## Activation() 

~~~C
nnom_layer_t* Activation(nnom_activation_t *act);	
~~~	

This is the Layer API wrapper for activations. It take activation instance as input.  

**Arguments**

- **act:** is the activation instance.

**Return**

- The activation layer instance

---

## Softmax() 

~~~C
nnom_layer_t* Softmax(void);
~~~

**Return**

- The Softmax layer instance

**Notes**

Softmax only has Layer API.


---

## ReLU() 

~~~C
nnom_layer_t* ReLU(void);
~~~

**Return**

- The ReLU layer instance

**Notes**

Using `layer = ReLU();` is no difference to `layer = Activation(act_relu());`

---

## Sigmoid() 

This function is affacted by an issue that we are currently working on. Check [issue](https://github.com/majianjia/nnom/issues/13)

~~~C
nnom_layer_t* Sigmoid(void);
~~~

**Return**

- The Sigmoid layer instance

**Notes**

Using `layer = Sigmoid();` is no difference to `layer = Activation(act_sigmoid());`

---

## TanH() 

~~~C
nnom_layer_t* TanH(void);
~~~

**Return**

- The TanH layer instance

This function is affacted by an issue that we are currently working on. Check [issue](https://github.com/majianjia/nnom/issues/13)

**Notes**

Using `layer = TanH();` is no difference to `layer = Activation(act_tanh());`

---


# Activation APIs


~~~C
nnom_activation_t* act_relu(void);
nnom_activation_t* act_sigmoid(void);
nnom_activation_t* act_tanh(void);
~~~

They return the activation instance which can be passed to either `model.active()` or `Activation()`

**Notes**

Softmax does not provided activation APIs. 

---

## Examples

** Using Layer's API **

Activation's layer API allows you to us them as a layer.
~~~C
nnom_layer_t layer;
model.add(&model, Dense(10));
model.add(&model, ReLU());
~~~

~~~C
nnom_layer_t layer;
model.add(&model, Dense(10));
model.add(&model, Activation(act_relu()));
~~~

~~~C
nnom_layer_t layer;
input = Input(shape(1, 10, 1), buffer);
layer = model.hook(Dense(10), input);
layer = model.hook(ReLU(), layer);
~~~

All 3 above perform the same and take the same memory.

** Using Activation's API **

~~~C
nnom_layer_t layer;
input = Input(shape(1, 10, 1), buffer);
layer = model.hook(Dense(10), input);
layer = model.active(act_relu(), layer);
~~~

This method perform the same but take less memory due to it uses the activation directly.
































