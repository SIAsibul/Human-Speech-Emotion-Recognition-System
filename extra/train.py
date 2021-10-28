import soundfile
import numpy as np
import librosa
import glob
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
import pickle


import pandas as pd
import re
import scipy.stats as stats
from scipy.io import wavfile
import numpy as np
import os

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





def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("data/GUBAS/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the gender label
        gender = basename.split("_")[0]
        # we allow only AVAILABLE_EMOTIONS we set
        # extract speech features
        frequencies = get_frequencies(file)
        features = None
        if len(frequencies) > 10:
            features = get_features(frequencies)
            # add to data
            X.append(features)
            y.append(gender)
            print(features)
            print(gender)
            global count
            count = count + 1
            print(count)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)


# load RAVDESS dataset
X_train, X_test, y_train, y_test = load_data(test_size=0.25)
# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted
# using utils.extract_features() method
print("[+] Number of features:", X_train.shape[1])
# best model, determined by a grid search
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 500,
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(X_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("../result"):
    os.mkdir("../result")

pickle.dump(model, open("../result/Gender/mlp_classifier_1.model", "wb"))
