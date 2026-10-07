import numpy as np
import sounddevice as sd

fs = 44100
t = np.linspace(0, 2, fs * 2, endpoint=False)
tone = np.sin(2*np.pi*440*t)

sd.play(tone, fs)
sd.wait()