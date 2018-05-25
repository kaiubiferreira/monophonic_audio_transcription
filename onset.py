import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import audio

def get_music_interval(input, window_size):
    window_index = 0
    current_energy = 0
    envelope = []
    for index, value in enumerate(input):
        if window_index < window_size:
            current_energy += value ** 2
            window_index += 1
        else:
            envelope = np.append(envelope, [current_energy / window_index], axis=0)
            window_index = 0
            current_energy = 0

    scaler = MinMaxScaler()
    envelope = scaler.fit_transform(envelope.reshape(-1, 1))

    start = 0
    end = len(envelope)

    for index, val in enumerate(envelope):
        if val < 0.001:
            print("eae")
        else:
            start = index
            break

    for index, val in reversed(list(enumerate(envelope))):
        if val < 0.001:
            print("eae")
        else:
            end = index
            break

    print(start, end)
    return (start, end)

def get_onset(array, duration, threshold=0.1):

    array = array.flatten()
    print(array.shape)
    print(array.ndim)

    peaks, _ = signal.find_peaks(array, height = 0.1, width=1)

    rolling_median = pd.rolling_median(array, 5) + threshold
    x_axis = np.linspace(0, duration, len(array))
    y_axis = array

    onset = []
    for index, value in enumerate(rolling_median):
        onset.append(array[index] if array[index] > value else 0)

    onset = np.array(onset)

    onset_filtered = []
    for index, value in enumerate(onset):
        if index == 0 or index >= len(onset) -1:
            onset_filtered.append(0)
        elif onset[index] <= onset[index - 1] or onset[index] <= onset[index + 1]:
            onset_filtered.append(0)
        else:
            onset_filtered.append(1)

    if plot_details:
        plt.subplot(4, 1, 1)
        plt.plot(x_axis, y_axis)
        plt.plot(x_axis, rolling_median)

        plt.subplot(4, 1, 2)
        plt.plot(x_axis, np.array(onset))

        plt.subplot(4, 1, 3)
        plt.plot(x_axis, onset_filtered)

        plt.subplot(4, 1, 4)
        plt.plot(array)
        plt.plot(peaks, array[peaks], 'bo')\

        plt.show()

    return onset_filtered


def plot_onset_preprocessing(array_list, duration):
    for index, array in enumerate(array_list):
        x_axis = np.linspace(0, duration, len(array))
        y_axis = array
        plt.subplot(len(array_list), 1, index + 1)
        plt.plot(x_axis, y_axis)

    plt.show()


def energy_derivative(Audio audio):
    window_index = 0
    current_energy = 0
    envelope = np.empty((1,))
    for index, value in enumerate(input):
        if window_index < window_size:
            current_energy += value ** 2
            window_index += 1
        else:
            envelope = np.append(envelope, [current_energy / window_index], axis=0)
            window_index = 0
            current_energy = 0

    derivative = np.gradient(envelope)
    derivative = np.array([0 if x < 0 else x for x in derivative])
    scaler = MinMaxScaler()
    envelope = scaler.fit_transform(envelope.reshape(-1, 1))
    derivative = scaler.fit_transform(derivative.reshape(-1, 1))
    derivative = np.log(derivative + 1)
    derivative = scaler.fit_transform(derivative.reshape(-1, 1))

    if plot_details:
        plot_onset_preprocessing([input, envelope, derivative], duration)

    get_onset(derivative, duration)

    return envelope, derivative

def get_energy(input):
    current_energy = 0
    for value in input:
        current_energy += value ** 2
    return current_energy

def hfc(input, window_size, duration, windosw_type=None):
    hfc_array = []
    hfc_current = 0

    for i, window in enumerate(sliding_window(input, window_size, 0)):
        window = window * signal.windows.hamming(window_size)
        transformed = abs(fftpack.fft(window))[0:int(len(window)/2)]

        hfc_previous = hfc_current
        hfc_current = np.sum(np.power(transformed, 2) * range(1, len(transformed) + 1))
        hfc_array.append(hfc_current)


    plt.plot(hfc_array)
    plt.show()
    fir = signal.firwin(11, 1.0 / 8, window="hamming")
    plt.plot(fir)
    plt.show()
    filtered = np.convolve(hfc_array, fir, mode="same")
    plt.plot(filtered)
    plt.show()
    derivative = np.gradient(filtered)
    derivative = np.array([0 if x < 0 else x for x in derivative])
    plt.plot(derivative)
    plt.show()


def sliding_window(input, window_size=1024, overlap=0):
    end = 0
    while end < len(input):
        start = max(end - overlap, 0)
        end = start + window_size
        if(end < len(input)):
            yield (input[start:end])