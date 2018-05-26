import numpy as np
import matplotlib.pyplot as plt


class Onset:

    def __init__(self, audio):
        self.audio = audio
        #self.adaptive_whitening()
        #self.spectral_flux()
        #plt.subplot(2, 1, 2)
        #plt.plot(self.hfc())
        #plt.show()

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

    def spectral_flux(self):

        previous = None
        flux = []
        for spectrum in self.audio.spectogram:
            if previous is None:
                flux.append(0)
            else:
                flux.append(np.sum(((spectrum - previous) + abs(spectrum - previous))/2))

            previous = spectrum

        flux = np.array(flux / max(flux))
        plt.plot(flux)
        plt.show()

    def adaptive_whitening(self, floor=5, relaxation=10):
        """
        "Adaptive Whitening For Improved Real-time Audio Onset Detection"
        Dan Stowel and Mark Plumbley (2007)
        """
        mem_coeff = 10.0 ** (-6. * relaxation / self.audio.rate)
        peak = None
        white_spectrum = []
        # iterate over all frames

        for spectrum in self.audio.spectogram:
            spec_floor = max(np.max(spectrum), floor)
            if peak is None:
                peak = spec_floor
            else:
                peak = max(spec_floor, mem_coeff * peak)

            white_spectrum.append(spectrum / peak)

        self.audio.plot_spectogram()
        print(self.audio.spectogram.shape)
        self.audio.spectogram = np.array(white_spectrum)
        print(self.audio.spectogram.shape)
        self.audio.plot_spectogram()
