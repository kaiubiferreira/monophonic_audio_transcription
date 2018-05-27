import numpy as np
import matplotlib.pyplot as plt
import pyreaper
from scipy.signal import find_peaks, medfilt, resample
from scipy import signal, fftpack


class Pitch:
    def __init__(self, audio):
        self.audio = audio
        self.f0_list = None
        self.set_pitch()

    def set_pitch(self):
        self.f0_list = self.autocorrelation()

    def get_pitch(self):
        return self.f0_list

    def plot_frequencies(self):
        x = np.linspace(0, self.audio.duration, len(self.f0_list))
        plt.plot(x, self.f0_list)
        plt.show()

    def window_to_sec(self, index):
        return self.audio.duration * index / len(self.f0_list)

    def autocorrelation(self):
        autocorr_list = [np.correlate(window, window, 'full') for window in self.audio.windows]

        f0_list = np.zeros(len(autocorr_list))

        for index, corr in enumerate(autocorr_list):
            peaks, _ = find_peaks(corr[int(corr.size / 2):], height=0)
            f0_list[index] = self.audio.rate / peaks[0]

        # high frequency filter
        f0_list = [0 if f0 > 4000 else f0 for f0 in f0_list]

        # median filter
        f0_list = medfilt(f0_list, 3)

        return f0_list

    def reaper(self):
        intWaveForm = np.ndarray.flatten(self.audio.waveform * 32000).astype('int16')
        pm_times, pm, f0_times, f0_list, corr = pyreaper.reaper(intWaveForm, self.audio.rate)

        return f0_list
