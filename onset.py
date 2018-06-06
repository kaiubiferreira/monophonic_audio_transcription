import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import madmom
import math


class Onset:

    def __init__(self, audio, algorithm='superflux'):
        self.audio = audio
        self.peaks = None
        self.odf = None
        self.algorithm = getattr(self, algorithm)
        self.set_onsets()

    def set_onsets(self):
        self.odf = self.algorithm()
        self.peaks = self.peak_picking()

    def get_onset(self):
        to_sec = (self.audio.num_windows / len(self.odf) * self.audio.window_size) / self.audio.rate
        return self.peaks * to_sec

    def plot_onset(self):
        plt.plot(self.odf)
        plt.plot(self.peaks, self.odf[self.peaks], 'bo')
        # plt.show()

    def get_sample_rate(self):
        return len(self.odf) / self.audio.duration

    def superflux(self):
        log_filt_spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(self.audio.file_path, num_bands=24)
        flux = madmom.features.onsets.superflux(log_filt_spec)
        flux = flux[:len(flux) - 5]
        flux = flux / max(flux)
        return flux

    def peak_picking_(self):
        return madmom.features.onsets.peak_picking(self.odf, threshold=0.2, pre_max=10, post_max=10)

    def peak_picking(self):
        rollin_window_size = 5
        minimum_distance = 5
        median = pd.Series(self.odf).rolling(window=rollin_window_size).mean() + 0.1
        median[:rollin_window_size-1] = median[rollin_window_size - 1]
        peaks = []
        last_index = None
        for index, value in enumerate(self.odf):
            if 0 < index < len(self.odf) - 1:
                if self.odf[index - 1] < self.odf[index] > self.odf[index + 1] and self.odf[index] > median[index]:
                    if last_index is None or index - last_index > minimum_distance:
                        peaks.append(index)
                        last_index = index

        plt.plot(self.odf)
        plt.plot(median)
        plt.plot(peaks, self.odf[peaks], 'bo')
        plt.show()

        return np.array(peaks)

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

    def superflux2(self):
        max_spectrogram = np.copy(self.audio.spectogram)
        for frame_index, frame in enumerate(self.audio.spectogram):
            for bin_index, bin in enumerate(frame):
                self.audio.spectogram[frame_index, bin_index] = math.log10(bin + 1)

        for frame_index, frame in enumerate(self.audio.spectogram):
            for bin_index, bin in enumerate(frame):
                if (0 < frame_index < len(self.audio.spectogram) - 1) and (0 < bin_index < len(frame)):
                    max_spectrogram[frame_index, bin_index] = max(self.audio.spectogram[frame_index - 1, bin_index],
                                                                  self.audio.spectogram[frame_index, bin_index],
                                                                  self.audio.spectogram[frame_index + 1, bin_index])
                else:
                    max_spectrogram[frame_index, bin_index] = 0

        flux = []
        u = 2
        for frame_index, frame in enumerate(self.audio.spectogram):
            if u <= frame_index < len(self.audio.spectogram) - u:
                flux.append(np.sum(((self.audio.spectogram[frame_index] - self.audio.spectogram[frame_index - u]) + abs(
                    self.audio.spectogram[frame_index] - self.audio.spectogram[frame_index - u])) / 2))
            else:
                flux.append(0)

        flux = np.array(flux / max(flux))

        return flux
