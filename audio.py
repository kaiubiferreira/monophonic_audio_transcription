import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


class Audio:

    def __init__(self, file_path, window_size=1024, overlap=0):
        # defines metadata
        self.file_path = file_path
        self.window_size = window_size
        self.overlap = overlap
        self.rate, raw = wavfile.read(file_path)
        self.num_channels = len(raw)

        # transforms to mono
        self.waveform = raw if self.num_channels == 1 else raw[:, 0] + raw[:, 1]
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

        # gets the frequency spectogram for all windows
        self.spec_frequency_axis, self.spec_time_axis, self.spectogram = signal.spectrogram(self.waveform,
                                                                                            self.rate,
                                                                                            window=signal.hann(
                                                                                                self.window_size),
                                                                                            nperseg=self.window_size,
                                                                                            mode='magnitude',
                                                                                            noverlap=self.overlap)

    def plot_waveform(self, downsampling_rate=0.25):
        plt.plot(signal.resample(self.waveform, int(self.num_frames * downsampling_rate)))
        plt.show()
        plt.plot(self.energy)
        plt.show()
        plt.pcolormesh(self.spec_time_axis, self.spec_frequency_axis, self.spectogram)
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