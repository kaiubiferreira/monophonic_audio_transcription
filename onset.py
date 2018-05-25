import numpy as np
import matplotlib.pyplot as plt


class Onset():

    def __init__(self, audio):
        self.audio = audio
        #plt.subplot(2, 1, 1)
        plt.plot(self.hfc())
        self.adaptive_whitening()
        #plt.subplot(2, 1, 2)
        plt.plot(self.hfc())
        plt.show()

    def energy_gradient(self):
        # gets the variation of energy
        gradient = np.gradient(self.audio.energy)

        # discards negative variations for onset detection, since we are interested in the increase of energy
        gradient = np.array([0 if x < 0 else x for x in gradient])

        # takes the logarithm to reduce the range and bring the outliers closer to the other points
        gradient = np.log(gradient + 1)

        # normalizes derivative
        gradient = gradient / max(gradient)

        return gradient

    def hfc(self):
        # disregard lower audible frequencies that might have high values and distort the hfc
        min_index = np.searchsorted(self.audio.spec_frequency_axis, [2000])[0]

        # calculates the HFC
        hfc_array = np.array([np.sum(np.abs(window[min_index:]) * np.arange(min_index, len(window))) for window in
                              np.transpose(self.audio.spectogram)])

        # gets the variation of HFC
        gradient = np.gradient(hfc_array)

        # discards negative variations for onset detection, since we are interested in the increase of energy
        gradient = np.array([0 if x < 0 else x for x in gradient])

        # normalizes gradient
        gradient = gradient / max(gradient)

        return gradient

    def adaptive_whitening(self, floor=5, relaxation=10):
        """
        "Adaptive Whitening For Improved Real-time Audio Onset Detection"
        Dan Stowel and Mark Plumbley (2007)
        """
        mem_coeff = 10.0 ** (-6. * relaxation / self.audio.rate)
        spectogram = np.transpose(self.audio.spectogram)
        peaks = []
        # iterate over all frames

        for window in range(self.audio.num_windows):
            spec_floor = max(np.max(spectogram[window]), floor)
            if window > 0:
                peaks.append(max(spec_floor, mem_coeff * peaks[window - 1]))
            else:
                peaks.append(spec_floor)

        self.audio.spectogram = self.audio.spectogram / peaks
