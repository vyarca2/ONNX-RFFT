import numpy as np
from scipy.io import wavfile

duration = 5
sample_rate = 16000
frequency = 440

t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

audio = np.sin(2 * np.pi * frequency * t)

audio *= 0.3 * np.iinfo(np.int16).max

audio = audio.astype(np.int16)

wavfile.write('/content/audio1.wav', sample_rate, audio)

print("Audio file saved as audio1.wav")
