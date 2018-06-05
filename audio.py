import matplotlib.pyplot as plt
import numpy as np
import math
import os
from scipy import signal
from scipy.io import wavfile


class Audio:

    def __init__(self, file_path, window_size=1024, overlap=0):
        # defines metadata
        self.file_path = file_path
        self.file_name = os.path.basename(file_path).split('.')[0]
        self.window_size = window_size
        self.overlap = overlap
        self.rate, raw = wavfile.read(file_path)
        self.num_channels = len(raw)

        # transforms to mono
        self.waveform = raw[:, 0] + raw[:, 1] if self.num_channels == 2 else raw
        self.num_frames = len(self.waveform)
        self.duration = self.num_frames / self.rate
        self.num_windows = int(self.num_frames / self.window_size)

        # normalize waveform
        self.waveform = self.waveform / max(self.waveform)

        # segments the input in windows
        self.windows = [np.array(self.waveform[frame:frame + self.window_size]) for frame in
                        range(1, self.num_frames, self.window_size - self.overlap) if
                        frame + self.window_size < self.num_frames]

        # calculates the energy vector
        self.energy = np.array([np.sum(window ** 2) for window in self.windows])

        # calculates the power level in Db
        self.power = np.array(
            [10 * math.log10((window / self.window_size) / (math.exp(-12))) for window in self.energy])

        # normalizes energy
        self.energy /= max(self.energy)

        # gets the frequency spectogram for all windows
        self.spec_frequency_axis, self.spec_time_axis, self.spectogram = signal.spectrogram(self.waveform,
                                                                                            self.rate,
                                                                                            window=signal.hann(
                                                                                                self.window_size),
                                                                                            nperseg=self.window_size,
                                                                                            mode='magnitude',
                                                                                            noverlap=self.overlap)
        self.spectogram = np.transpose(self.spectogram)

    def filter_silence(self):
        for sample_index, sample in enumerate(self.waveform):
            window_index = int(sample_index/self.window_size)
            if window_index < len(self.energy) and window_index < len(self.power):
                if self.energy[window_index] < 0.001 or self.power[window_index] < 0:
                    self.waveform[sample_index] = self.energy[window_index] = self.power[window_index] = 0
            else:
                self.waveform[sample_index] = 0

    def plot_waveform(self, downsampling_rate=0.125):
        plt.plot(signal.resample(self.waveform, int(self.num_frames * downsampling_rate)))
        plt.show()

    def plot_spectogram(self):
        plt.pcolormesh(self.spec_time_axis, self.spec_frequency_axis, np.transpose(self.spectogram))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    def describe(self):
        print("-----------------------------------------------")
        print("File Path..........: {0}".format(self.file_path))
        print("Duration...........: {0}s".format(self.duration))
        print("Sampling Rate......: {0}Hz".format(self.rate))
        print("Window Size........: {0}".format(self.num_windows))
        print("Number of frames...: {0}".format(self.num_frames))
        print("Number of windows..: {0}".format(self.num_windows))
        print("-----------------------------------------------")
