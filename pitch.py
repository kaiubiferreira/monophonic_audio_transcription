import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt

class Pitch:
    def __init__(self, audio):
        self.audio = audio
        self.autocorrelation()

    def autocorrelation(self):
        autocorr_list = [np.correlate(window, window, 'full') for window in self.audio.windows]

        f0_list = np.zeros(len(autocorr_list))

        for index, corr in enumerate(autocorr_list):
            peaks, _ = find_peaks(corr[int(corr.size/2):], height=0)
            f0_list[index] = self.audio.rate / peaks[0]

        # high frequency filter
        f0_list = [0 if f0 > 4000 else f0 for f0 in f0_list]

        # median filter
        f0_list = medfilt(f0_list, 3)

        return f0_list