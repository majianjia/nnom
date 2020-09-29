"""
This file is a part of noise deduction example of NNoM
https://github.com/majianjia/nnom

The dataset generation is based on MS-SNSD dataset(https://github.com/microsoft/MS-SNSD) which contains clean speech and noise.
The MS-SNSD can mix clean speech and noise to generate 3 parts of data: noisy speech, clean speech, and noise with coded
filenames.

In this script, we will use the MS-SNSD dataset to generate the training data of our NN.
"""
from python_speech_features import mfcc, fbank, hz2mel, mel2hz
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os
import re

def get_band_filter_coeff(samplerate, f0, Q=1.0):
    """
    Bandpass filter based on BLT: Cookbook formulae for audio EQ biquad filter coefficients
    https://gist.github.com/RyanMarcus/d3386baa6b4cb1ac47f4#file-gistfile1-txt
    """
    w0 = 2 * np.pi * f0 / samplerate
    alpha = np.sin(w0) / (2 * Q)
    a = np.zeros(3)
    b = np.zeros(3)
    b[0] = Q*alpha
    b[1] = 0
    b[2] = -Q*alpha
    a[0] = 1 + alpha
    a[1] = -2*np.cos(w0)
    a[2] = 1-alpha
    return  b, a

def iir_design_first_order(band_frequency, samplerate, normalize=True): # the ban frequency is the middel fre
    b = []
    a = []
    for i in range(len(band_frequency)):
        b_, a_ = get_band_filter_coeff(samplerate, band_frequency[i])
        if(normalize):
            b_ = b_/a_[0]           # unified
            a_[1:] = a_[1:]/a_[0]
            a_[0] = 1
        b.append(b_)
        a.append(a_)
    return b, a
    # Ref implementation:
    # b, a = set_gains(b_in, a_in, alpha, gains[0])
    # i = 0
    # g = 0
    # for n in range(2, len(x)):
    #     y[n] = b[0] * x[n] + b[1] * x[n - 1] + b[2] * x[n - 2] - a[1]* y[n - 1] - a[2] * y[n - 2]
    #     if (n % step == 0 and i < len(gains)-1):
    #         i += 1
    #         g = gains[i] * 0.4 + g*0.6
    #         b, a = set_gains(b_in, a_in, alpha, g)
    # return y

def generate_filter_header(b, a, order, filename='equalizer_coeff.h'):
    def array2str(data):
        s = np.array2string(np.array(data).flatten(), separator=',')
        return s.replace("\n", "").replace("\r", "").replace(' ', '').replace(',', ', ').replace('[', '{').replace(']', '}')
    with open(filename, 'w') as file:
        file.write("\n#define NUM_FILTER " + str(len(b)) + '\n')
        file.write("\n#define NUM_ORDER " +  str(order) + '\n')
        file.write("\n#define NUM_COEFF_PAIR " + str(order*2+1) + '\n')
        file.write("\n#define FILTER_COEFF_A " + array2str(a) + "\n")
        file.write("\n#define FILTER_COEFF_B " + array2str(b) + "\n")

def iir_design(band_frequency, samplerate, order=1): # the ban frequency is the middel fre
    b = []
    a = []
    fre = band_frequency / (samplerate/2)
    for i in range(1, len(band_frequency)-1):
        b_, a_ = signal.iirfilter(order, [fre[i] - (fre[i]-fre[i-1])/2, fre[i]+ (fre[i+1]-fre[i])/2],
                                  btype='bandpass', output='ba')
        # b_, a_ = signal.iirfilter(order, [fre[i-1], fre[i+1]-0.001],
        #                            btype='bandpass', output='ba')
        # b_, a_ = signal.cheby1(order, 1, [fre[i] - (fre[i]-fre[i-1])/2, fre[i]+ (fre[i+1]-fre[i])/2],
        #                           btype='bandpass', output='ba')
        b.append(b_)
        a.append(a_)
    return b, a

def fir_design(band_frequency, samplerate, order=51):
    from scipy import signal
    b = []
    fre = band_frequency / (samplerate/2)
    for i in range(1, len(band_frequency)-1):
        b.append(signal.firwin(order, [fre[i] - (fre[i]-fre[i-1])/2, fre[i]+ (fre[i+1]-fre[i])/2], pass_zero='bandpass'))
    return b

def get_mel_scale(nfilt=20, samplerate=16000, lowfreq=20, highfreq=8000):
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    return melpoints

def bandpass_filter_fir(sig, b_in, a_in, step, gains):
    from scipy import signal
    x = sig
    y = np.zeros(len(x))
    state = np.zeros(len(b_in)-1)
    g=0
    for n in range(0, len(gains)):
        g = max(0.8*g, gains[n])    # pre RNNoise paper https://arxiv.org/pdf/1709.08243.pdf
        b = b_in * g
        filtered, state = signal.lfilter(b, 1, x[n*step: min((n+1)*step, len(x))], zi=state)
        y[n*step: min((n+1)*step, len(x))] = filtered
    return y

def bandpass_filter_iir(sig, b_in, a_in, step, gains):
    from scipy import signal
    x = sig
    y = np.zeros(len(x))
    state = np.zeros(len(b_in)-1)
    g=0
    for n in range(0, len(gains)):
        g = max(0.6*g, gains[n])    # r=0.6 pre RNNoise paper https://arxiv.org/pdf/1709.08243.pdf
        b = b_in*g
        a = a_in
        filtered, state = signal.lfilter(b, a, x[n*step: min((n+1)*step, len(x))], zi=state)
        y[n*step: min((n+1)*step, len(x))] = filtered
    return y


def plot_frequency_respond(b, a=None, fs=16000):
    a = a if len(a) == len(b)  else np.ones(len(b))
    for i in range(len(b)):
        w, h = signal.freqz(b[i], a[i])
        plt.plot(w*0.15915494327*fs, 20 * np.log10(np.maximum(abs(h), 1e-5)), 'b')
    plt.title('Digital filter frequency response')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [Hz]')
    plt.show()

def noise_suppressed_example(plot=False):
    """
    In this example, we demonstrate how we suppress noise using dynamic gains in an audio equalizer [EQ].
    The basic idea is we use the clean to noisy energy ratio of each frequency band as the gain of suppression.
    It is done in a very small windows (500 point = 31.25ms) so that it can respone very quickly.
    Then we apply these gains to an equalizer (a set of parallel bandpass filter). The gains are changing very fast
    so the noise will be suppressed when it is detected.

    This is also the principle that how do we generate the truth gains for the training data (y_train).
    """
    # change here to select the file and its noise mixing level.
    nfilt = 20
    test_num = 1          # which file
    test_noise_level = 10  # noise level in db, selected from 0, 10, 20, depeneded on dataset

    # change here to select the file and its noise mixing level.
    clean_file = "MS-SNSD/CleanSpeech_training/clnsp" + str(test_num) + ".wav"
    noisy_file = "MS-SNSD/NoisySpeech_training/noisy"+str(test_num)+"_SNRdb_"+str(test_noise_level)+".0_clnsp"+str(test_num) +".wav"

    (rate, clean_sig) = wav.read(clean_file)
    (rate, noisy_sig) = wav.read(noisy_file)
    clean_sig = clean_sig/32768
    noisy_sig = noisy_sig/32768

    # Calculate the energy of each frequency bands
    clean_band_eng, _ = fbank(clean_sig, rate, winlen=0.032, winstep=0.032/2, nfilt=nfilt, nfft=512, lowfreq=20, highfreq=8000, preemph=0)
    noisy_band_eng, _ = fbank(noisy_sig, rate, winlen=0.032, winstep=0.032/2, nfilt=nfilt, nfft=512, lowfreq=20, highfreq=8000, preemph=0)
    # gains
    gains = np.sqrt(clean_band_eng / noisy_band_eng)
    if(plot):
        plt.title("Gains")
        plt.plot(gains[:, :10])
        plt.show()

    # convert mel scale back to frequency band
    mel_scale = get_mel_scale(nfilt=nfilt, lowfreq=20, highfreq=8000)
    band_freq = mel2hz(mel_scale)
    band_frequency = band_freq[1:-1] # the middle point of each band
    print('band frequency', band_frequency)

    # the noisy audio now pass to a set of parallel band pass filter.
    # which performed like an audio equalizer [EQ]
    # the different is we will change the gains of each band very quickly so that we suppress the noise while keeping the speech.
    # design our band pass filter for each band in the equalizer.
    # becasue the frequency band is overlapping, we need to reduce the signal to avoid overflow when converting back to int16.

    print("denoising using IIR filter")
    b, a = iir_design(band_freq, rate)
    if plot:
        plot_frequency_respond(b, a)
    print("b", b)
    print("a", a)
    step = int(0.03125 * rate / 2)
    print("audio process step:", step)
    filtered_signal = np.zeros(len(noisy_sig))
    for i in range(len(b)):
        filtered_signal += bandpass_filter_iir(noisy_sig, b[i].copy(), a[i].copy(), step, gains[:, i])
        print("filtering with frequency: ", band_frequency[i])
    filtered_signal = filtered_signal * 0.6

    filtered_signal = np.clip(filtered_signal, -1, 1)
    wav.write("_filtered_sample.wav", rate, np.asarray(filtered_signal * 32767, dtype=np.int16))
    wav.write("_noisy_sample.wav", rate, np.asarray(noisy_sig * 32767, dtype=np.int16))
    print("noisy signal is saved to:", "_noisy_sample.wav")
    print("filtered signal is saved to:", "_filtered_sample.wav")


def generate_data(path, vad_active_delay=0.07, vad_threshold=1e-1, random_volume=True, winlen=0.032, winstep=0.032/2,
                  numcep=13, nfilt=26, nfft=512, lowfreq=20, highfreq=8000, winfunc=np.hanning, ceplifter=0,
                  preemph=0.97, appendEnergy=True):
    """
    vad_filter_size: number of winstep for filter. if one of the point is active, the first size/2 and last size/2 will be actived
    Larger size will have better cover to the speech, but will bring none-speech moments
    please refer to python_speech_features.mfcc for other parameters
    """
    mfcc_data = []
    filename_label = []
    total_energy = []
    band_energy = []
    vad = []
    files = os.listdir(path)
    for f in files:
        filename = f
        if ('wav' not in filename):
            continue
        (rate, sig) = wav.read(path+'/'+f)
        # convert file to [-1, 1)
        sig = sig/32768

        # calculate the energy per band, this was one of the step in mfcc but taked out
        band_eng, total_eng = fbank(sig, rate, winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft, lowfreq=lowfreq,
                                   highfreq=highfreq, preemph=preemph, winfunc=winfunc)

        # for the mfcc, because we are not normalizing them,
        # so we randomize the volume to simulate the real life voice record.
        if(random_volume):
            sig = sig * np.random.uniform(0.8, 1)

        # calculate mfcc features
        mfcc_feat = mfcc(sig, rate, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft,
                         lowfreq=lowfreq, highfreq=highfreq, winfunc=winfunc, ceplifter=ceplifter, preemph=preemph,
                         appendEnergy=appendEnergy)

        # voice active detections, only valid with clean speech. Detected by total energy vs threshold.
        v = (total_eng > vad_threshold).astype(int)
        vad_delay = int(vad_active_delay*(rate*winstep))
        conv_win = np.concatenate([np.zeros(vad_delay), np.ones(vad_delay)]) # delay the VAD for a vad_active_delay second
        v = np.convolve(v, conv_win, mode='same')
        v = (v > 0).astype(int)

        total_energy.append(total_eng)
        band_energy.append(band_eng)
        vad.append(v)
        mfcc_data.append(mfcc_feat.astype('float32'))
        filename_label.append(filename)
    return mfcc_data, filename_label, total_energy, vad, band_energy


if __name__ == "__main__":
    # This example will generate 2 files, noisy speech and noise suppressed speech.
    # You might open them with your player to get a feeling ot what does it sound like.
    # It give you an idea that how does this energy based noise suppression work.
    noise_suppressed_example()

    # change this will change the whole system, including equalizer and RNN
    # it set: number of filter in equalizer, number of mfcc feature, and number of RNN output.
    # choose from 10 ~ 30.
    num_filter = 20

    # generate filter coefficient
    mel_scale = get_mel_scale(nfilt=num_filter, lowfreq=20, highfreq=8000)
    band_freq = mel2hz(mel_scale)
    b, a = iir_design(band_freq, 16000, order=1) # >2 order will not stable with only float32 accuracy in C.
    generate_filter_header(b, a, order=int(b[0].shape[-1] / 2), filename='equalizer_coeff.h')
    # plot frequency respond
    #plot_frequency_respond(b, a)

    print('Reading noisy and clean speech files...')
    # dataset generation start from here:
    # energy thresehold for voice activivity detection in clean speech.
    vad_energy_threashold = 0.1

    noisy_speech_dir = 'MS-SNSD/NoisySpeech_training'
    clean_speech_dir = 'MS-SNSD/CleanSpeech_training'
    noise_dir = 'MS-SNSD/Noise_training'

    # clean sound, mfcc, and vad
    print('generating clean speech MFCC...')
    clean_speech_mfcc, clean_file_label, total_energy, vad, clnsp_band_energy = \
        generate_data(clean_speech_dir, nfilt=num_filter, numcep=num_filter, appendEnergy=True, preemph=0, vad_threshold=vad_energy_threashold)

    # add noise to clean speech, then generate the noise MFCC
    print('generating noisy speech MFCC...')
    noisy_speech_mfcc, noisy_file_label, _, _ , noisy_band_energy= \
        generate_data(noisy_speech_dir, nfilt=num_filter, numcep=num_filter, appendEnergy=True, preemph=0, vad_threshold=vad_energy_threashold)

    # MFCC for noise only
    print('generating noisy MFCC...')
    noise_only_mfcc, noise_only_label, _, _ , noise_band_energy= \
        generate_data(noise_dir, random_volume=False, nfilt=num_filter, numcep=num_filter, appendEnergy=True, preemph=0)

    # plt.plot(vad[5], label='voice active')
    # plt.plot(total_energy[5], label='energy')
    # plt.legend()
    # plt.show()

    # combine them together
    clnsp_mfcc = []
    noisy_mfcc = []
    noise_mfcc = []
    voice_active = []
    gains_array = []

    print('Processing training data')
    for idx_nosiy, label in enumerate(noisy_file_label):
        # get file encode from file name e.g. "noisy614_SNRdb_30.0_clnsp614.wav"
        nums = re.findall(r'\d+', label)
        file_code = nums[0]
        db_code = nums[1]

        # get clean sound name
        idx_clnsp = clean_file_label.index('clnsp'+str(file_code)+'.wav')

        # truth gains y_train
        gains = np.sqrt(clnsp_band_energy[idx_clnsp]/ noisy_band_energy[idx_nosiy])
        #gains = clnsp_band_energy[idx_clnsp] / noisy_band_energy[idx_nosiy]
        gains = np.clip(gains, 0, 1)

        # experimential, suppress the gains when there is no voice detected
        #gains[vad[idx_clnsp] < 1] = gains[vad[idx_clnsp] < 1] / 10
        # g = np.swapaxes(gains, 0, 1)
        # plt.imshow(g, interpolation='nearest', origin='lower', aspect='auto')
        # plt.show()

        # get all data needed
        voice_active.append(vad[idx_clnsp])
        clnsp_mfcc.append(clean_speech_mfcc[idx_clnsp])
        noisy_mfcc.append(noisy_speech_mfcc[idx_nosiy])
        noise_mfcc.append(noise_only_mfcc[idx_nosiy]) # noise has the same index as noisy speech
        gains_array.append(gains)

        #>>> Uncomment to plot the MFCC image
        # mfcc_feat1 = np.swapaxes(clean_speech_mfcc[idx_clnsp], 0, 1)
        # mfcc_feat2 = np.swapaxes(noisy_speech_mfcc[idx_nosiy], 0, 1)
        # fig, ax = plt.subplots(2)
        # ax[0].set_title('MFCC Audio:' + str(idx_clnsp))
        # ax[0].imshow(mfcc_feat1, origin='lower', aspect='auto', vmin=-8, vmax=8)
        # ax[1].imshow(mfcc_feat2, origin='lower', aspect='auto', vmin=-8, vmax=8)
        # plt.show()

    # save the dataset.
    np.savez("dataset.npz", clnsp_mfcc=clnsp_mfcc, noisy_mfcc=noisy_mfcc, noise_mfcc=noise_mfcc, vad=voice_active, gains=gains_array)
    print("Dataset generation has been saved to:", "dataset.npz")