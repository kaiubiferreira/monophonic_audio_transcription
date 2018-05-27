import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Music:
    def __init__(self, audio, onset, pitch):
        self.onset = onset
        self.pitch = pitch
        self.audio = audio
        self.notes = pd.DataFrame(columns=['f0', 'duration'], index=['note_index'])
        # self.plot_pitch_onset()
        self.pitch_filter()
        # self.plot_pitch_onset()
        self.note_filter()
        # self.plot_pitch_onset()

    def pitch_filter(self):
        pitch_array = self.pitch.get_pitch()
        onset_array = self.onset.get_onset()
        current_note_pitch = []
        current_onset_index = -1
        filtered_pitch = np.array([])
        note_index = 1

        for i in np.arange(0, len(pitch_array)):
            current_note_pitch.append(pitch_array[i])
            current_time = self.pitch.window_to_sec(i)

            if current_onset_index + 1 < len(onset_array) and current_time > onset_array[current_onset_index + 1]:
                median = self.get_median(current_note_pitch)
                curr_array = np.repeat(median, len(current_note_pitch))
                filtered_pitch = np.concatenate((filtered_pitch, curr_array))
                self.notes = self.notes.append(
                    pd.DataFrame({'f0': median, 'duration': self.pitch.window_to_sec(len(current_note_pitch))},
                                 index=[note_index]))

                note_index = note_index + 1
                current_note_pitch = []
                current_onset_index = current_onset_index + 1

        curr_array = np.repeat(np.median(current_note_pitch), len(current_note_pitch))
        filtered_pitch = np.concatenate((filtered_pitch, curr_array))
        self.notes = self.notes.append(
            pd.DataFrame({'f0': median, 'duration': self.pitch.window_to_sec(len(current_note_pitch))},
                         index=[note_index]))

        self.notes = self.notes.dropna()
        self.pitch.f0_list = filtered_pitch

    def plot_pitch_onset(self):
        x = np.linspace(0, self.audio.duration, len(self.pitch.get_pitch()))
        plt.plot(x, self.pitch.get_pitch())
        for line in self.onset.get_onset():
            plt.axvline(line, color='green')
        plt.show()

    def get_median(self, array, border_percent=0.10):
        return np.median(array[int(len(array) * border_percent): int(len(array) * (1 - border_percent))])

    def note_filter(self):
        frequencyTable = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87, 32.70,
                          34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30,
                          73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.8, 110.0, 116.5, 123.5, 130.8, 138.6, 146.8,
                          155.6, 164.8, 174.6, 185.0, 196.0, 207.7, 220.0, 233.1, 246.9, 261.6, 277.2, 293.7, 311.1,
                          329.6, 349.2, 370.0, 392.0, 415.3, 440.0, 466.2, 493.9, 523.3, 554.4, 587.3, 622.3, 659.3,
                          698.5, 740.0, 784.0, 830.6, 880.0, 932.3, 987.8, 1047, 1109, 1175, 1245, 1319, 1397, 1480,
                          1568, 1661, 1760, 1865, 1976, 2093, 2217, 2349, 2489, 2637, 2794, 2960, 3136, 3322, 3520,
                          3729, 3951, 4186, 4435, 4699, 4978, 5274, 5588, 5920, 6272, 6645, 7040, 7459, 7902]
        dict_names = {
            0: {'step': 'C', 'accidental': 'NATURAL'},
            1: {'step': 'C', 'accidental': 'SHARP'},
            2: {'step': 'D', 'accidental': 'NATURAL'},
            3: {'step': 'E', 'accidental': 'FLAT'},
            4: {'step': 'E', 'accidental': 'NATURAL'},
            5: {'step': 'F', 'accidental': 'NATURAL'},
            6: {'step': 'F', 'accidental': 'SHARP'},
            7: {'step': 'G', 'accidental': 'NATURAL'},
            8: {'step': 'G', 'accidental': 'SHARP'},
            9: {'step': 'A', 'accidental': 'NATURAL'},
            10: {'step': 'B', 'accidental': 'FLAT'},
            11: {'step': 'B', 'accidental': 'NATURAL'},
        }

        minimum = 100
        maximum = 4000

        self.pitch.f0_list = [
            frequencyTable[np.searchsorted(frequencyTable, int(f0))] if int(f0) in range(minimum, maximum) else 0
            for f0 in self.pitch.f0_list]

        for index, row in self.notes.iterrows():
            if minimum < row.f0 < maximum:
                frequency_index = np.searchsorted(frequencyTable, int(row.f0))
                self.notes.at[index, 'f1'] = frequencyTable[frequency_index]
                self.notes.at[index, 'step'] = dict_names[frequency_index % 12]['step']
                self.notes.at[index, 'accidental'] = dict_names[frequency_index % 12]['accidental']
                self.notes.at[index, 'octave'] = int(frequency_index / 12)

            else:
                self.notes.at[index, 'f1'] = '0'
                self.notes.at[index, 'step'] = 'pause'
                self.notes.at[index, 'accidental'] = None
                self.notes.at[index, 'octave'] = None
