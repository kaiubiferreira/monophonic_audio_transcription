from audio import Audio
from onset import Onset
from pitch import Pitch
from music import Music

audio = Audio('audio/spring.wav')
onset = Onset(audio, algorithm='superflux2')

pitch = Pitch(audio, algorithm='reaper')
music = Music(audio, onset, pitch)
music.write_file(folder='musicxml')