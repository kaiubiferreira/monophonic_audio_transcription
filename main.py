from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from audio import Audio
import onset
from onset import Onset
from pitch import Pitch

sns.set()

def plot_wave_form(signal, duration, sub):
    print("plotting {0} frames".format(len(signal)))
    x_axis = np.linspace(0, duration, len(signal))
    y_axis = signal

    plt.subplot(3, 1, sub)
    plt.plot(x_axis, y_axis)


audio = Audio('audio/spring_mono.wav')
#audio.plot_waveform()
#onset.hfc(audio)
#onset.adaptive_whitening(audio)
onset = Onset(audio)
onset.get_onsets()
# x = onset.energy_gradient()
# print(audio.num_windows)
# print(len(audio.energy))
# print(len(x))
#onset.get_onsets()
#pitch = Pitch(audio)


#rate, data = wavfile.read('audio/spring.wav')
#first_channel = data[:, 0]
#duration = len(first_channel) / rate

#first_channel = resample(first_channel, int(len(first_channel)/2))
#plot_wave_form(first_channel, duration)

##envelope, derivative = onset.energy_derivative(input=first_channel, window_size=1024, duration=duration)
#onset.hfc(input=first_channel, window_size=512, duration=duration)