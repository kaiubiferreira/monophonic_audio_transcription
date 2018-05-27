import numpy as np
import matplotlib.pyplot as plt
import madmom


class Onset:

    def __init__(self, audio):
        self.audio = audio
        self.peaks = None
        self.odf = None
        self.set_onsets()

    def set_onsets(self):
        self.odf = self.superflux()
        self.peaks = self.peak_picking()

    def get_onset(self):
        to_sec = (self.audio.num_windows / len(self.odf) * self.audio.window_size) / self.audio.rate
        return self.peaks * to_sec

    def plot_onset(self):
        plt.plot(self.odf)
        plt.plot(self.peaks, self.odf[self.peaks], 'bo')
        plt.show()

    def superflux(self):
        log_filt_spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(self.audio.file_path, num_bands=24)
        flux = madmom.features.onsets.superflux(log_filt_spec)
        flux = flux[:len(flux) - 5]
        flux = flux / max(flux)
        return flux

    def peak_picking(self):
        return madmom.features.onsets.peak_picking(self.odf, threshold=0.2, pre_max=10, post_max=10)
        # median = pd.rolling_median(odf, 7) + threshold
        # filtered = np.array([0 if x < median[i] else x for i, x in enumerate(odf)])
        # peaks = np.array([i for i, x in enumerate(filtered) if x > 0])
        # peaks = np.array([peaks[index] for index, difference in enumerate(np.abs(peaks - np.roll(peaks, 1)))
        #                    if difference > 3])
        #
        # plt.plot(filtered)
        # plt.plot(peaks, filtered[peaks], 'bo')
        # plt.show()
        #
        # return peaks

        # print(list(onsets))
        # print(list(np.roll(onsets, 1)))
        # print(list(onsets - np.roll(onsets, 1)))

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
                              self.audio.spectogram])

        # plt.plot(hfc_array)
        # plt.show()

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
                flux.append(np.sum(((spectrum - previous) + abs(spectrum - previous)) / 2))

            previous = spectrum

        flux = np.array(flux / max(flux))
        return flux

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
