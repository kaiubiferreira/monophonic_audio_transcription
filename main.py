from audio import Audio
from onset import Onset
from pitch import Pitch
from music import Music

audio = Audio('audio/pirates.wav')
onset = Onset(audio, algorithm='superflux')
pitch = Pitch(audio, algorithm='reaper')
music = Music(audio, onset, pitch)
music.write_file(folder='musicxml')