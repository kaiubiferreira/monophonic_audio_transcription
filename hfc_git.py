import scipy
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# References:
# Bello, Daudet, Abdallah, Duxbury, Davies, Sandler: A Tutorial on Onset Detection in Music Signals

def detect_onsets(sig, fftwin = 512):
    spectrogram = generate_spectrogram(sig, fftwin)
    hfcs = [get_hfc(spectrum) for spectrum in spectrogram]

    plt.plot(spectrogram[51])
    plt.show()

    hfcs = hfcs/max(hfcs)

    hfcs = filter_hfcs(hfcs)
    plt.plot(hfcs)

    plt.show()

    peak_indices = np.array([i for i, x in enumerate(hfcs) if x > 0]) * fftwin
    return peak_indices
    #return hfcs

def get_hfc(spectrum):
    if(spectrum is not None):
        hfc = np.sum(np.power(spectrum, 2) * np.arange(1, len(spectrum) + 1))
    else:
        hfc = 0
    return hfc

def generate_spectrogram(audio, window_size):
    spectrogram = [None] * int((1 + (len(audio) / window_size)))
    for t in range(0, len(audio), window_size):
        actual_window_size = min(window_size, len(audio) - t)
        windowed_signal = audio[t:(t + window_size)] * np.hanning(actual_window_size)
        spectrum = abs(scipy.fft(windowed_signal))
        spectrum = spectrum[0:int(len(spectrum) / 2)]
        spectrogram[int(t / window_size)] = spectrum

    return spectrogram

def filter_hfcs(hfcs):
    fir = signal.firwin(11, 1.0 / 8, window = "hamming")
    plt.plot(fir)
    plt.show()
    filtered = np.convolve(hfcs, fir, mode="same")
    filtered = climb_hills(filtered)
    plt.plot(filtered)
    plt.show()
    #filtered = climb_hills(hfcs)
    return filtered

def climb_hills(vector):
    moving_points = list(range(len(vector)))
    stable_points = []

    while len(moving_points) > 0:
        for (i, x) in reversed(list(enumerate(moving_points))):

            def stable():
                stable_points.append(x)
                del moving_points[i]

            if x > 0 and x < len(vector) - 1:
                if vector[x] >= vector[x - 1] and vector[x] >= vector[x + 1]:
                    stable()
                elif vector[x] < vector[x - 1]:
                    moving_points[i] -= 1
                else:
                    moving_points[i] += 1

            elif x == 0:
                if vector[x] >= vector[x + 1]:
                    stable()
                else:
                    moving_points[i] += 1

            else:
                if vector[x] >= vector[x - 1]:
                    stable()
                else:
                    moving_points[i] -= 1

    filtered = [0] * len(vector)
    for x in set(stable_points):
        filtered[x] = vector[x]

    return filtered

import scipy.io.wavfile
sr, audio = scipy.io.wavfile.read('audio/spring.wav')
audio = audio[:,0] # make it mono
onsets = detect_onsets(audio)

for el in onsets:
    print(onsets)