import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from lxml import etree


def quantize(x, precision=2, base=.05):
    return round(base * round(float(x) / base), precision)


class Music:
    def __init__(self, audio, onset, pitch):
        self.onset = onset
        self.pitch = pitch
        self.audio = audio
        self.notes = pd.DataFrame(columns=['f0', 'duration_seconds'], index=['note_index'])
        self.adjusted_notes = pd.DataFrame(columns=['measure_id', 'type', 'step', 'octave', 'accidental', 'is_rest',
                                                    'start_tie', 'end_tie', 'duration'])
        self.bpm = 0
        self.measure_size = 0
        self.division = 32
        self.xml = ""

        self.pitch_filter()
        self.note_filter()
        self.set_tempo()
        self.set_beats()
        self.set_measure_size()
        self.divide_notes()
        self.set_xml()

    def write_file(self, folder=""):
        if folder != "":
            folder = folder + "/"

        output_file_name = folder + self.audio.file_name + ".xml"
        with open(output_file_name, "w") as f:
            f.write(self.xml)
            print("musicxml saved to file " + output_file_name)

    def set_xml(self):
        root = etree.Element("score-partwise")
        work = etree.SubElement(root, "work")
        etree.SubElement(work, "work-title").text = self.audio.file_name
        identification = etree.SubElement(root, "identification")
        encoding = etree.SubElement(identification, "encoding")
        etree.SubElement(encoding, "software").text = "Monophonic Music Transcription"
        defaults = etree.SubElement(root, "defaults")
        scaling = etree.SubElement(defaults, "scaling")
        etree.SubElement(scaling, "millimeters").text = "7.0"
        etree.SubElement(scaling, "tenths").text = "40"
        page_layout = etree.SubElement(defaults, "page-layout")
        etree.SubElement(page_layout, "page-height").text = "1700.00"
        etree.SubElement(page_layout, "page-width").text = "1190.00"
        credit = etree.SubElement(root, "credit", page="1")
        credit_words = etree.SubElement(credit, "credit-words", justify="center",
                                        valign="top")
        credit_words.text = self.audio.file_name
        credit_words.attrib["font-size"] = "22"
        part_list = etree.SubElement(root, "part-list")
        score_part = etree.SubElement(part_list, "score-part", id="P1")
        etree.SubElement(score_part, "part-name").text = "Viola"
        etree.SubElement(score_part, "part-abbreviation").text = "Vla."
        instrument = etree.SubElement(score_part, "score-instrument", id="P1-I3")
        etree.SubElement(instrument, "instrument-name").text = "Viola"
        midi_instrument = etree.SubElement(score_part, "midi-instrument", id="P1-I3")
        etree.SubElement(midi_instrument, "midi-channel").text = "1"
        etree.SubElement(midi_instrument, "midi-program").text = "42"
        etree.SubElement(midi_instrument, "volume").text = "78.7402"
        etree.SubElement(midi_instrument, "pan").text = "0"

        part = etree.SubElement(root, "part", id="P1")
        previous_measure = 0
        for row in self.adjusted_notes.iterrows():
            current_measure = row[1]['measure_id']
            if current_measure != previous_measure:
                measure = etree.SubElement(part, "measure", number=str(current_measure))

                # measure metadata
                if current_measure == 1:
                    attributes = etree.SubElement(measure, "attributes")
                    key = etree.SubElement(attributes, "key")
                    etree.SubElement(key, "fifths").text = "0"
                    etree.SubElement(key, "mode").text = "major"
                    time = etree.SubElement(attributes, "time")
                    etree.SubElement(time, "beats").text = str(int(self.measure_size))
                    etree.SubElement(time, "beat-type").text = "4"
                    clef = etree.SubElement(attributes, "clef")
                    etree.SubElement(clef, "sign").text = "C"
                    etree.SubElement(clef, "line").text = "3"
                    etree.SubElement(measure, "sound", tempo=str(self.bpm))

                previous_measure = current_measure

            note = etree.SubElement(measure, "note")

            if row[1]['is_rest']:
                etree.SubElement(note, "rest")
                etree.SubElement(note, "duration").text = str(int(row[1]['duration']))
            else:
                pitch = etree.SubElement(note, "pitch")
                etree.SubElement(pitch, "step").text = str(row[1]['step'])
                if row[1]['accidental'] == 'FLAT':
                    etree.SubElement(pitch, "alter").text = "-1"
                elif row[1]['accidental'] == 'SHARP':
                    etree.SubElement(pitch, "alter").text = "1"

                etree.SubElement(note, "duration").text = str(row[1]['duration'])
                if row[1]['start_tie']:
                    etree.SubElement(note, "tie", type="start")
                if row[1]['end_tie']:
                    etree.SubElement(note, "tie", type="stop")

                etree.SubElement(pitch, "octave").text = str(int(row[1]['octave']))
                etree.SubElement(note, "voice").text = "1"

            etree.SubElement(note, "type").text = row[1]['type']

            if row[1]['start_tie'] or row[1]['end_tie']:
                notations = etree.SubElement(note, "notations")

            if row[1]['start_tie']:
                etree.SubElement(notations, "tied", type="start")

            if row[1]['end_tie']:
                etree.SubElement(notations, "tied", type="stop")

        self.xml = etree.tostring(root, pretty_print=True, encoding='unicode',
                                  doctype='<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE score-partwise PUBLIC '
                                          '"-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
                                          '"http://www.musicxml.org/dtds/partwise.dtd">')

    def divide_notes(self):
        # self.measure_size = 1.5
        end_tie = False
        remaining_space = self.measure_size
        measure_id = 1
        for row in self.notes.iterrows():
            current_beat = row[1]['adjusted_beats']

            while current_beat > 0:
                if current_beat < remaining_space:
                    self.insert_note(row, current_beat, measure_id, end_tie=end_tie)
                    remaining_space -= current_beat
                    end_tie = False
                    current_beat = 0
                elif current_beat == remaining_space:
                    self.insert_note(row, current_beat, measure_id, end_tie=end_tie)
                    remaining_space = self.measure_size
                    current_beat = 0
                    measure_id = measure_id + 1
                else:
                    self.insert_note(row, remaining_space, measure_id, start_tie=True, end_tie=end_tie)
                    end_tie = True
                    current_beat -= remaining_space
                    remaining_space = self.measure_size
                    measure_id = measure_id + 1

    def insert_note(self, row, beats, measure_id, start_tie=False, end_tie=False):
        figure_dict = {
            0.125: '32nd',
            0.25: '16th',
            0.5: 'eighth',
            1.0: 'quarter',
            2.0: 'half',
            4.0: 'whole'
        }
        keys = list(figure_dict.keys())

        inserted_beats = 0

        while inserted_beats < beats:
            idx = np.searchsorted(keys, beats - inserted_beats, side='left')

            if idx >= len(keys) or (idx > 0 and keys[idx] > beats):
                current_beat = keys[idx - 1]
            else:
                current_beat = keys[idx]

            if inserted_beats + current_beat < beats:
                start_tie = True

            self.adjusted_notes = self.adjusted_notes.append({
                'measure_id': measure_id,
                'type': figure_dict[current_beat],
                'step': row[1]['step'],
                'octave': row[1]['octave'],
                'accidental': row[1]['accidental'],
                'is_rest': True if row[1]['step'] == 'pause' else False,
                'start_tie': start_tie,
                'end_tie': end_tie,
                'duration': current_beat * self.division
            }, ignore_index=True)

            inserted_beats = inserted_beats + current_beat
            if start_tie == True:
                end_tie = True
            start_tie = False

    def set_beats(self):
        self.notes['beats'] = self.notes['duration_seconds'] * (self.bpm / 60)
        self.notes['adjusted_beats'] = self.notes['beats'].apply(lambda x: quantize(x, 2, 0.25))

    def set_measure_size(self):
        # finds the measure size that minimizes the number of connected notes between measures
        best_measure_size = None
        best_measure_tie_count = None

        for size in range(2, 6):
            tie_count = 0
            remaining_space = size
            for beat in self.notes['adjusted_beats']:
                current_beat = beat

                while current_beat > 0:
                    if current_beat < remaining_space:
                        remaining_space -= current_beat
                        current_beat = 0
                    elif current_beat == remaining_space:
                        remaining_space = size
                        current_beat = 0
                    else:
                        current_beat -= remaining_space
                        remaining_space = size
                        tie_count += 1

            if best_measure_tie_count is None or tie_count < best_measure_tie_count:
                best_measure_tie_count = tie_count
                best_measure_size = size

        self.measure_size = best_measure_size

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
                    pd.DataFrame({'f0': median, 'duration_seconds': self.pitch.window_to_sec(len(current_note_pitch))},
                                 index=[note_index]))

                note_index = note_index + 1
                current_note_pitch = []
                current_onset_index = current_onset_index + 1

        curr_array = np.repeat(np.median(current_note_pitch), len(current_note_pitch))
        filtered_pitch = np.concatenate((filtered_pitch, curr_array))
        self.notes = self.notes.append(
            pd.DataFrame({'f0': median, 'duration_seconds': self.pitch.window_to_sec(len(current_note_pitch))},
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
                self.notes.at[index, 'accidental'] = ''
                self.notes.at[index, 'octave'] = None

    def set_tempo(self):
        window_size = 512
        sample_count = len(self.onset.odf)
        window_count = math.ceil(sample_count / window_size)

        # append zeros
        oss = np.concatenate((self.onset.odf, np.zeros(int(window_count * window_size - sample_count))))

        # separate windows
        oss = oss.reshape(math.ceil(sample_count / window_size), window_size)
        # print(oss.shape)

        hm = pd.Series(dtype=np.dtype('float64'))
        for sample in oss:
            fft_sample = sample
            corr = np.correlate(fft_sample, fft_sample, 'full')
            corr = corr[int(corr.size / 2):]

            b_hat_previous = None
            b_hat_count = 0
            am_sum = 0
            for t, am in enumerate(corr):
                if t != 0:
                    b = 60 * self.onset.get_sample_rate() / t
                    b_hat = quantize(x=b, precision=2, base=1.0)

                    if b_hat_previous is not None and b_hat != b_hat_previous:
                        if b_hat_previous not in hm.keys():
                            hm.append(pd.Series([0.0], index=[b_hat_previous]))
                            hm[b_hat_previous] = 0.0

                        hm[b_hat_previous] = float(hm[b_hat_previous] + am_sum / b_hat_count)
                        b_hat_count = 0.0
                        am_sum = 0.0

                    am_sum += am
                    b_hat_count += 1
                    b_hat_previous = b_hat

        hm = hm[(hm.index > 50) & (hm.index < 180)]

        self.bpm = hm.idxmax()
        # print(hm.idxmax())
        # hm = hm.sort_index()
        # plt.plot(hm)
        # plt.show()
