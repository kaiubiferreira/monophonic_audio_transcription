from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from audio import Audio
from onset import Onset
from pitch import Pitch
from music import Music

sns.set()

def plot_wave_form(signal, duration, sub):
    print("plotting {0} frames".format(len(signal)))
    x_axis = np.linspace(0, duration, len(signal))
    y_axis = signal

    plt.subplot(3, 1, sub)
    plt.plot(x_axis, y_axis)


audio = Audio('audio/mario_mono.wav')
onset = Onset(audio)
pitch = Pitch(audio)
music = Music(audio, onset, pitch)
