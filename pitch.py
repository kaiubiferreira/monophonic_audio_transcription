import numpy as np
import matplotlib.pyplot as plt
import pyreaper
from scipy.signal import find_peaks, medfilt
from scipy import signal, fftpack

class Pitch:
    def __init__(self, audio):
        self.audio = audio
        autocorr = self.autocorrelation()
        reaper = self.reaper()
        x = np.linspace(0, self.audio.duration, len(autocorr))
        plt.plot(x, autocorr)
        x = np.linspace(0, self.audio.duration, len(reaper))
        plt.plot(x, reaper)

        plt.show()

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


