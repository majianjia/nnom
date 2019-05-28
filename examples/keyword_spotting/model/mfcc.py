

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import random

def load_noise(path='dat/_background_noise_/'):
    noise = []
    files = os.listdir(path)
    for f in files:
        filename = f
        if ('wav' not in filename):
            continue
        f = os.path.join(path, f)
        (rate, sig) = wav.read(f)
        noise.append(sig)
    return  noise

def generate_mfcc(sig, rate, sig_len, noise=None, noise_weight=0.1, winlen=0.03125, winstep=0.03125/2, numcep=13, nfilt=26, nfft=512, lowfreq=20, highfreq=4000, winfunc=np.hanning, ceplifter=0, preemph=0.97):
    if(len(sig) != sig_len):
        if(len(sig)< sig_len):
            sig = np.pad(sig, (0, sig_len - len(sig)), 'constant')
        if(len(sig) >sig_len):
            sig = sig[0:sig_len]
    # i dont know, 'tensorflow' normalization
    sig = sig.astype('float') / 32768

    if(noise is not None):
        noise = noise[random.randint(0, len(noise)-1)] # pick a noise
        start = random.randint(0, len(noise)-sig_len) # pick a sequence
        noise = noise[start:start+sig_len]
        noise = noise.astype('float')/32768
        sig = sig * (1-noise_weight) + noise * noise_weight
        #wav.write('noise_test.wav', rate, sig)
    mfcc_feat = mfcc(sig, rate, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft, lowfreq=lowfreq,
                     highfreq=highfreq, winfunc=winfunc, ceplifter=ceplifter, preemph=preemph)
    mfcc_feat = mfcc_feat.astype('float32')
    return mfcc_feat

def merge_mfcc_file(input_path='dat/', mix_noise=True, sig_len=16000, winlen=0.03125, winstep=0.03125/2, numcep=13, nfilt=26, nfft=512,
                    lowfreq=20, highfreq=4000, winfunc=np.hanning, ceplifter=0, preemph=0.97):

    train_data = []
    test_data = []
    validate_data = []
    train_lable = []
    test_label = []
    validate_label =[]

    if mix_noise:
        noise = load_noise()
    else:
        noise = None

    with open(input_path + 'testing_list.txt', 'r') as f:
        test_list = f.read()
    with open(input_path +  'validation_list.txt', 'r') as f:
        validate_list = f.read()

    files = os.listdir(input_path)
    for fi in files:
        fi_d = os.path.join(input_path, fi)
        # folders of each cmd
        if os.path.isdir(fi_d):
            label = fi_d.split('/')[1] # get the label from the dir
            print(label)
            # noise in training
            if 'noise' in label:
                for f in os.listdir(fi_d):
                    filename = f
                    if('wav' not in filename):
                        continue
                    f = os.path.join(fi_d, f)
                    (rate, sig) = wav.read(f)
                    for i in range(0, len(sig), sig_len):
                        data = generate_mfcc(sig[i:i+sig_len], rate, sig_len, winlen=winlen, winstep=winstep, numcep=numcep,
                                             nfilt=nfilt, nfft=nfft, lowfreq=lowfreq,
                                             highfreq=highfreq, winfunc=winfunc, ceplifter=ceplifter, preemph=preemph)
                        data = np.array(data)  # ?? no idea why this works
                        train_data.append(data)
                        train_lable.append('noise')

                continue
            # dataset
            for f in os.listdir(fi_d):
                filename = f
                f = os.path.join(fi_d, f)
                (rate, sig) = wav.read(f)
                data = generate_mfcc(sig, rate, sig_len, noise=noise, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft, lowfreq=lowfreq,
                     highfreq=highfreq, winfunc=winfunc, ceplifter=ceplifter, preemph=preemph)
                data = np.array(data) # ?? no idea why this works

                # split dataset into train, test, validate
                if filename in test_list:
                    test_data.append(data)
                    test_label.append(label)
                elif filename in validate_list:
                    validate_data.append(data)
                    validate_label.append(label)
                else:
                    train_data.append(data)
                    train_lable.append(label)

    # finalize
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    validate_data = np.array(validate_data)

    return (train_data, train_lable), (test_data, test_label), (validate_data, validate_label)


if __name__ == "__main__":

    # test
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = merge_mfcc_file()

    np.save('train_data.npy', x_train)
    np.save('train_label.npy', y_train)
    np.save('test_data.npy', x_test)
    np.save('test_label.npy', y_test)
    np.save('val_data.npy', x_val)
    np.save('val_label.npy', y_val)

    print('x_train shape:', x_train.shape, 'max', x_train.max(), 'min', x_train.min())

    mfcc_feat = x_train[3948]
    mfcc_feat = np.swapaxes(mfcc_feat, 0, 1)
    ig, ax = plt.subplots()
    cax = ax.imshow(mfcc_feat, interpolation='nearest', origin='lower', aspect='auto')
    ax.set_title('MFCC')
    plt.show()
