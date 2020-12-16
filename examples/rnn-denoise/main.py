
import matplotlib.pyplot as plt
import os
from tensorflow.keras import *
from tensorflow.keras  import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model, save_model
from scipy import signal
import tensorflow as tf
import numpy as np

import sys
sys.path.append(os.path.abspath("../../scripts"))
print(sys.path)

from gen_dataset import *
from nnom import *

def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
     return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)


def filter_voice(sig, rate, gains, nband=26, lowfreq=20, highfreq=8000):
    # see gen_dataset.py's example for detial
    mel_scale = get_mel_scale(nfilt=nband, lowfreq=lowfreq, highfreq=highfreq)
    band_freq = mel2hz(mel_scale)
    band_frequency = band_freq[1:-1] # the middle point of each band
    print('band frequency', band_frequency)
    b, a = iir_design(band_freq, rate)
    step = int(0.032 * rate / 2)
    filtered_signal = np.zeros(len(sig))
    for i in range(len(b)):
        filtered_signal += bandpass_filter_iir(sig, b[i].copy(), a[i].copy(), step, gains[:, i])
        print("filtering with frequency: ", band_frequency[i])
    filtered_signal =filtered_signal * 0.6
    return filtered_signal

def normalize(data, n, quantize=True):
    limit = pow(2, n)
    data = np.clip(data, -limit, limit)/limit
    if quantize:
        data = np.round(data * 128)/ 128.0
    return data

def voice_denoise(sig, rate, model, timestamp_size, numcep=26, plot=False):
    sig = sig / 32768
    # get the mfcc of noisy voice
    mfcc_feat = mfcc(sig, rate, winlen=0.032, winstep=0.032/2, numcep=numcep, nfilt=numcep, nfft=512,
                     lowfreq=20, highfreq=8000, winfunc=np.hanning, ceplifter=0, preemph=0, appendEnergy=True)
    mfcc_feat = mfcc_feat.astype('float32')

    # differential of mfcc, add 0 to the beginning
    diff = np.diff(mfcc_feat, axis=0)
    diff = np.concatenate([[mfcc_feat[0]], diff], axis=0)  # first derivative
    diff1 = np.diff(diff, axis=0)
    diff1 = np.concatenate([[diff[0]], diff1], axis=0) # second derivative
    diff = diff[:, :10]
    diff1 = diff1[:, :10]

    # concat both differential and original mfcc
    feat = np.concatenate([mfcc_feat, diff, diff1], axis=-1)

    # requantise the MFCC (same as training data)
    feat = normalize(feat, 3, quantize=False)
    # plt.hist(feat.flatten(), bins=1000)
    # plt.show()

    # interference.
    feat = np.reshape(feat, (feat.shape[0], 1, feat.shape[1]))
    feat = feat[: feat.shape[0] // timestamp_size * timestamp_size]
    prediction = model.predict(feat, batch_size=timestamp_size)
    if(type(prediction) is list):
        predicted_gains = prediction[0]
        predicted_vad = prediction[1]
    else:
        predicted_gains = prediction
        predicted_vad = None

    # now process the signal.
    filtered_sig = filter_voice(sig, rate=rate, gains=predicted_gains, nband=mfcc_feat.shape[-1])
    if(plot):
        for i in range(10):
            plt.plot(predicted_gains[:, i], label='band'+str(i))
        if(predicted_vad is not None):
            plt.plot(predicted_vad, 'r', label='VAD')
        plt.ylabel("Gains")
        plt.xlabel("MFCC Sample")
        plt.legend()
        plt.show()
    return filtered_sig

# differential of mfcc, add 0 to the beginning
def get_diff_list(data):
    L = []
    for d in data:
        L.append(np.concatenate([[d[0]], np.diff(d, axis=-2)], axis=-2))
    return np.array(L)

# we need to reset state in RNN. becasue we dont each batch are different. however, we need statful=true for nnom
class reset_state_after_batch(tf.keras.callbacks.Callback):
    reset_after = 1 # reset state after N batch.
    curr = 0
    def on_batch_end(self, batch, logs=None):
        self.curr += 1
        if(self.curr >= self.reset_after):
            self.curr = 0
            self.model.reset_states()
        pass

def train_simple(x_train, y_train, vad_train, batch_size=64, epochs=10, model_name="model.h5"):
    """
    This simple RNN model also can do similar jobs. Compared to the complex RNNoise-like model:
    it take the same input as the other one, but train with only the gains (without VAD)
    it also have a simple straight forward structure (no concatenate).
    """
    input_feature_size = x_train.shape[-1]
    output_feature_size = y_train.shape[-1]
    timestamp_size = batch_size

    input = Input(shape=(1, input_feature_size), batch_size=timestamp_size)

    x = GRU(96, return_sequences=True, stateful=True, recurrent_dropout=0.3)(input)
    x = GRU(96, return_sequences=True, stateful=True, recurrent_dropout=0.3)(x)
    x = GRU(48, return_sequences=True, stateful=True, recurrent_dropout=0.3)(x)
    x = Flatten()(x)
    x = Dense(output_feature_size)(x)
    x = Activation("hard_sigmoid")(x) # use hard sigmoid for better resolution in fixed-point model

    model = Model(inputs=input, outputs=[x])
    model.compile("adam", loss=["MSE"], metrics=[msse])
    model.summary()
    history = model.fit(x_train, y_train,
                        batch_size=timestamp_size, epochs=epochs, verbose=2, shuffle=False, # shuffle must be false
                        callbacks=[reset_state_after_batch()])
    # free the session to avoid nesting naming while we load the best model after.
    save_model(model, model_name)
    del model
    tf.keras.backend.clear_session()
    return history

def train(x_train, y_train, vad_train, batch_size=64, epochs=10, model_name="model.h5"):
    """
    RNNoise-like structure with some adaption to fit NNoM's implementation.
    """
    input_feature_size = x_train.shape[-1]
    output_feature_size = y_train.shape[-1]
    timestamp_size = batch_size
    input = Input(shape=(1, input_feature_size), batch_size=timestamp_size)

    """
        This is an RNNoise-like structure
    """
    # voice activity detection
    x1_1 = GRU(24, return_sequences=True, stateful=True, recurrent_dropout=0.2)(input)
    x1_1 = Dropout(0.3)(x1_1)
    x1_2 = GRU(24, return_sequences=True, stateful=True, recurrent_dropout=0.2)(x1_1)
    x1_2 = Dropout(0.3)(x1_2)
    x = Flatten()(x1_2)
    x = Dropout(0.3)(x)
    x = Dense(1)(x)
    vad_output = Activation("hard_sigmoid")(x)

    # we dont concate input with layer output, because the range different will cause quite many quantisation lost.
    x_in = GRU(64, return_sequences=True, stateful=True, recurrent_dropout=0.3)(input)

    # Noise spectral estimation
    x2 = concatenate([x_in, x1_1, x1_2], axis=-1)
    x2 = GRU(48, return_sequences=True, stateful=True, recurrent_dropout=0.3)(x2)
    x2 = Dropout(0.3)(x2)

    #Spectral subtraction
    x3 = concatenate([x_in, x2, x1_2], axis=-1)
    x3 = GRU(96, return_sequences=True, stateful=True, recurrent_dropout=0.3)(x3)
    x3 = Dropout(0.3)(x3)
    x = Flatten()(x3)
    x = Dense(output_feature_size)(x)
    x = Activation("hard_sigmoid")(x)

    """
        Simplified RNNoise-Like model. 
    """
    # x = GRU(64, return_sequences=True, stateful=True)(input)
    # x2 = GRU(24, return_sequences=True, stateful=True)(x)
    # x3 = Flatten()(x2)
    # x3 = Dense(1)(x3)
    # vad_output = Activation("hard_sigmoid")(x3)
    # x = GRU(48, return_sequences=True, stateful=True)(x)
    # x = concatenate([x, x2])
    # x = GRU(48, return_sequences=True, stateful=True)(x)
    # x = Flatten()(x)
    # x = ReLU()(x)
    # x = Dense(output_feature_size)(x)
    # x = Activation("hard_sigmoid")(x) # use hard sigmoid for better resolution

    model = Model(inputs=input, outputs=[x, vad_output])
    #model.compile("adam", loss=[mycost, my_crossentropy], loss_weights=[10, 0.5], metrics=[msse])  # RNNoise loss and cost
    model.compile("adam", loss=["MSE", my_crossentropy], loss_weights=[10, 2], metrics=[msse])
    model.summary()

    history = model.fit(x_train, [y_train, vad_train],
                        batch_size=timestamp_size, epochs=epochs, verbose=2, shuffle=False, # shuffle must be false
                        callbacks=[reset_state_after_batch()])# validation_split=0.1)

    # free the session to avoid nesting naming while we load the best model after.
    save_model(model, model_name)
    del model
    tf.keras.backend.clear_session()
    return history

def main():
    # load test dataset. Generate by gen_dataset.py see the file for details.
    try:
        dataset = np.load('dataset.npz', allow_pickle=True)
    except:
        raise Exception("dataset.npz not found, please run 'gen_dataset.py' to create dataset")

    # combine them together
    clnsp_mfcc = dataset['clnsp_mfcc']    # mfcc
    noisy_mfcc = dataset['noisy_mfcc']
    vad = dataset['vad']                  # voice active detection
    gains = dataset['gains']              # gains

    # get mfcc derivative from dataset.
    clnsp_mfcc_diff = get_diff_list(clnsp_mfcc)
    noisy_mfcc_diff = get_diff_list(noisy_mfcc)
    clnsp_mfcc_diff1 = get_diff_list(clnsp_mfcc_diff)
    noisy_mfcc_diff1 = get_diff_list(noisy_mfcc_diff)

    # combine all pices to one large array
    clnsp_mfcc = np.concatenate(clnsp_mfcc, axis=0)
    noisy_mfcc = np.concatenate(noisy_mfcc, axis=0)
    clnsp_mfcc_diff = np.concatenate(clnsp_mfcc_diff, axis=0)
    noisy_mfcc_diff = np.concatenate(noisy_mfcc_diff, axis=0)
    clnsp_mfcc_diff1 = np.concatenate(clnsp_mfcc_diff1, axis=0)
    noisy_mfcc_diff1 = np.concatenate(noisy_mfcc_diff1, axis=0)
    vad = np.concatenate(vad, axis=0)
    gains = np.concatenate(gains, axis=0)

    # these max and min are rear
    print('mfcc max:', noisy_mfcc.max(), 'mfcc min:', noisy_mfcc.min())
    print('mfcc diff max:', noisy_mfcc_diff.max(), 'mfcc diff min:', noisy_mfcc_diff.min())

    # preprocess data
    timestamp_size = 2048 # this must be > than 1024, since we are using 1 one sample as a batch, which still too small for BP
    num_sequence = len(vad) // timestamp_size
    print('timestamp', timestamp_size, 'num of data', num_sequence)

    # prepare data
    diff = np.copy(noisy_mfcc_diff[:num_sequence * timestamp_size, :10])
    diff1 = np.copy(noisy_mfcc_diff1[:num_sequence * timestamp_size, :10])
    feat = np.copy(noisy_mfcc[:num_sequence * timestamp_size, :])

    # concat mfcc, 1st and 2nd derivative together as the training data.
    x_train = np.concatenate([feat, diff, diff1], axis=-1)
    # convert MFCC range to -1 to 1.0 In quantization, we will saturate them to leave more resolution in smaller numbers
    # we saturate the peak to leave some more resolution in other band.
    x_train = normalize(x_train, 3, quantize=False)
    # plt.hist(gains.flatten(), bins=1000)
    # plt.show()

    # reshape
    x_train = np.copy(x_train[:num_sequence * timestamp_size, :])
    x_train = np.reshape(x_train, (num_sequence* timestamp_size, 1, x_train.shape[-1]))
    y_train = np.copy(gains[:num_sequence * timestamp_size,:])
    y_train = np.reshape(y_train, (num_sequence* timestamp_size, gains.shape[-1]))
    vad_train = np.copy(vad[:num_sequence * timestamp_size]).astype(np.float32)
    vad_train = np.reshape(vad_train, (num_sequence * timestamp_size, 1))

    # train the model, choose either one.
    history = train(x_train, y_train, vad_train, batch_size=timestamp_size, epochs=5, model_name="model.h5")
    #history = train_simple(x_train, y_train, vad_train, batch_size=timestamp_size, epochs=10, model_name="model.h5")

    # get the model
    model = load_model("model.h5", custom_objects={'mycost': mycost, 'msse':msse, 'my_crossentropy':my_crossentropy, 'my_accuracy':my_accuracy})

    # denoise a file for test.
    # Make sure the MFCC parameters inside the voice_denoise() are the same as our gen_dataset.
    (rate, sig) = wav.read("_noisy_sample.wav")
    filtered_sig = voice_denoise(sig, rate, model, timestamp_size, numcep=y_train.shape[-1], plot=True) # use plot=True argument to see the gains/vad
    wav.write("_nn_filtered_sample.wav", rate, np.asarray(filtered_sig * 32767, dtype=np.int16))

    # now generate the NNoM model
    generate_model(model, x_train[:timestamp_size*4], name='denoise_weights.h')
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if(physical_devices is not None):
       tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main()

    # def convert_to_inference_model(original_model):
    #     """ https://gist.github.com/rpicatoste/02cecac1ed52524301e3ab423dac888b """
    #     import json
    #     from tensorflow.keras.models import model_from_json
    #     original_model_json = original_model.to_json()
    #     inference_model_dict = json.loads(original_model_json)
    #     model = inference_model_dict['config']
    #     for layer in model['layers']:
    #         if 'stateful' in layer['config']:
    #             layer['config']['stateful'] = True
    #
    #         if 'batch_input_shape' in layer['config']:
    #             layer['config']['batch_input_shape'][0] = 1
    #             layer['config']['batch_input_shape'][1] = 1
    #
    #     inference_model = model_from_json(json.dumps(inference_model_dict))
    #     inference_model.set_weights(original_model.get_weights())
    #     del original_model
    #     return inference_model
    #
    # model = convert_to_inference_model(model)
    # save_model(model, "model.h5")
    # del model
    # tf.keras.backend.clear_session()
    # model = load_model("model.h5", custom_objects={'mycost': mycost, 'msse': msse, 'my_crossentropy': my_crossentropy,
    #                                                'my_accuracy': my_accuracy})
