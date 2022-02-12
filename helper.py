import wave
from sys import byteorder
from array import array
from struct import pack
import soundfile
import numpy as np
import librosa


import scipy.stats as stats
from scipy.io import wavfile
import numpy as np
import os
import pickle
THRESHOLD = 500
# CHUNK_SIZE = 1024
# FORMAT = pyaudio.paInt16
RATE = 16000
#
# SILENCE = 30

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result


def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r



count = 0

def get_features(frequencies):
    print("\nExtracting features ")
    nobs, minmax, mean, variance, skew, kurtosis = stats.describe(frequencies)
    median = np.median(frequencies)
    mode = stats.mode(frequencies).mode[0]
    std = np.std(frequencies)
    low, peak = minmax
    q75, q25 = np.percentile(frequencies, [75 ,25])
    iqr = q75 - q25
    return nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr


def get_frequencies(file):
    rate, data = wavfile.read(file)
    # get dominating frequencies in sliding windows of 200ms
    step = rate/5 # 3200 sampling points every 1/5 sec
    step = int(step)
    window_frequencies = []
    frequency = None
    for i in range(0, len(data), step):
        ft = np.fft.fft(data[i:i+step])  # fft returns the list N complex numbers
        freqs = np.fft.fftfreq(len(ft))  # fftq tells you the frequencies associated with the coefficients
        imax = np.argmax(np.abs(ft))
        freq = freqs[imax]
        freq_in_hz = abs(freq * rate)
        window_frequencies.append(freq_in_hz)
        filtered_frequencies = [f for f in window_frequencies if 20<f<280 and not 46<f<66]  # I see noise at 50Hz and 60Hz
    frequency = filtered_frequencies

    return frequency
