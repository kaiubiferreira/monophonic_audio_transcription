import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from audio import Audio


def energy_gradient(audio):
    # gets the variation of energy
    gradient = np.gradient(audio.energy)

    # discards negative variations for onset detection, since we are interested in the increase of energy
    gradient = np.array([0 if x < 0 else x for x in gradient])

    # log
    gradient = np.log(gradient + 1)

    # normalizes derivative
    gradient = gradient / max(gradient)

    plt.plot(gradient)
    plt.show()
    return gradient


def hfc(audio):
    print(audio.spectogram.shape)
    print(np.transpose(audio.spectogram).shape)
    print(audio.spec_frequency_axis.shape)
    print(audio.spec_time_axis.shape)
    # print(audio.spec_frequency_axis)
    # audio.plot_waveform()

    min_index = np.searchsorted(audio.spec_frequency_axis, [3000])[0]
    print(audio.spec_frequency_axis[min_index])

    hfc = np.array([np.sum(np.square(window[min_index:]) * np.arange(min_index, len(window)))  for window in np.transpose(audio.spectogram)])
    print("hfc shape:")
    print(hfc.shape)

    hfc = hfc/max(hfc)
    plt.subplot(2, 1, 1)
    plt.plot(hfc)
    #plt.show()

    fir = signal.firwin(11, 1.0 / 8, window="hamming")
    filtered = np.convolve(hfc, fir, mode="same")
    plt.subplot(2, 1, 2)
    plt.plot(filtered)
    plt.show()

    masri = [0]
    for i in range(1, audio.num_windows - 1):
        masri.append((hfc[i]/hfc[i-1])*(hfc[i]/audio.energy[i]))

    plt.plot(masri)
    plt.show()

    gradient = np.gradient(hfc)
    gradient = np.array([0 if x < 0 else x for x in gradient])
    plt.plot(gradient)
    plt.show()

# def hfc(input, window_size, duration, windosw_type=None):
#     hfc_array = []
#     hfc_current = 0
#
#     for i, window in enumerate(sliding_window(input, window_size, 0)):
#         window = window * signal.windows.hamming(window_size)
#         transformed = abs(fftpack.fft(window))[0:int(len(window)/2)]
#
#         hfc_previous = hfc_current
#         hfc_current = np.sum(np.power(transformed, 2) * range(1, len(transformed) + 1))
#         hfc_array.append(hfc_current)
#
#
#     plt.plot(hfc_array)
#     plt.show()
#     fir = signal.firwin(11, 1.0 / 8, window="hamming")
#     plt.plot(fir)
#     plt.show()
#     filtered = np.convolve(hfc_array, fir, mode="same")
#     plt.plot(filtered)
#     plt.show()
#     derivative = np.gradient(filtered)
#     derivative = np.array([0 if x < 0 else x for x in derivative])
#     plt.plot(derivative)
#     plt.show()
