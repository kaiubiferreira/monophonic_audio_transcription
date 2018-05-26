import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from scipy import signal, fftpack

class Pitch:
    def __init__(self, audio):
        self.audio = audio
        #autocorr = self.autocorrelation()
        plt.plot(self.autocorrelation())
        plt.plot(self.cepstrum())
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

    def cepstrum(self):
        f0_list = []
        for win in self.audio.windows:
            fft_result = np.log(np.square(fftpack.rfft(win * signal.hann(self.audio.window_size))))
            ceps = np.square(fftpack.ifft(fft_result).real)
            ceps = ceps[20:int(len(ceps) / 2)]
            f0_list.append(self.audio.rate / (np.argmax(ceps)))

        #plt.plot(f0_list)
        #plt.show()
        return f0_list
        #print(np.argmax(ceps))
        #print(self.audio.rate / np.argmax(ceps))

        # plt.subplot(3, 1, 1)
        # plt.plot(self.audio.windows[35])
        # plt.subplot(3, 1, 2)
        # plt.plot(fft_result)
        # plt.subplot(3, 1, 3)
        # plt.plot(ceps)
        #
        # #plt.subplot(3, 1, 3)
        # #plt.plot(cepstrum)
        # plt.show()