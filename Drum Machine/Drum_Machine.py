# Drum machine usage:
# - Generates a repeating, 16-step drum loop with synthesized kick, snare, and hi-hat sounds.
# - Install dependencies from a terminal with: py -m pip install numpy sounddevice
# - This will have to be run locally, as the online environment does not support audio output or keyboard input.
# - Edit kick_pattern, snare_pattern, and hihat_pattern below to change the rhythm:
#   each 1 plays that drum on a step; each 0 leaves the step silent.
# - Change the bpm argument in the sequencer call near the bottom to adjust tempo.
# - Make sure an audio output device is available; press Ctrl+C in the terminal to stop.

import numpy as np
import sounddevice as sd
import time 

SAMPLE_RATE = 44100

def sin(frequency, duration, phase=0.0):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t + phase)
   
def fade_out(sound, fade_ms=10):
    fade_samples = int(fade_ms * SAMPLE_RATE / 1000)
    fade = np.linspace(1, 0, fade_samples)
    sound[-fade_samples:] *= fade
    sound[-1] = 0.0 
    return sound
   
def kick(duration=0.5, f_start=100.0, f_end=40.0):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    k = np.log(f_start / f_end) / duration
    freq_t = f_start * np.exp(-k * t)

    phase = 2 * np.pi * np.cumsum(freq_t) / SAMPLE_RATE
    raw = np.sin(phase)

    decay = np.exp(-8 * t)
    raw = fade_out(raw, fade_ms=10)
    return raw * decay

def snare_bp(duration=0.25):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    noise = np.random.normal(0, 1, len(t))

    hp = noise - np.convolve(noise, np.ones(12)/12, mode='same')

    bp = np.convolve(hp, np.ones(6)/6, mode='same')

    tone = np.sin(2 * np.pi * 200 * t)

    mix = 0.7 * bp + 0.3 * tone

    env = np.exp(-18 * t)

    return mix * env


def hi_hat(duration=0.05):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    noise = np.random.normal(0, 1, len(t))
    noise_hp = noise - np.convolve(noise, np.ones(32)/32, mode='same')

    freqs = [4000, 6000, 8000, 10000, 12000, 15000]
    metal = np.zeros_like(t)
    for f in freqs:
        metal += np.sign(np.sin(2 * np.pi * f * t))
    metal /= len(freqs)

    mix = .3*(0.6 * noise_hp + 0.4 * metal)

    env = np.exp(-100 * t)
    hat = mix * env

    fade_samples = int(0.003 * SAMPLE_RATE)
    fade = np.linspace(1, 0, fade_samples)
    hat[-fade_samples:] *= fade
    hat[-1] = 0.0

    return hat



def loop_audio(buffer):
    idx = 0
    length = len(buffer)

    def callback(outdata, frames, time, status):
        nonlocal idx
        for i in range(frames):
            outdata[i] = buffer[idx]
            idx = (idx + 1) % length  

    return sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=callback
    )


kick_pattern  = [1,0,0,0, 0,0,0,0, 1,0,1,0, 0,0,0,0]
snare_pattern = [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0]
hihat_pattern = [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1]

def sequencer(kick_pattern, snare_pattern, hihat_pattern, bpm=120):
    steps = len(kick_pattern)
    step_duration = 60 / bpm / 4
    step_samples = int(step_duration * SAMPLE_RATE)

    loop_length = steps * step_samples
    extra = int(0.1 * SAMPLE_RATE)  
    output = np.zeros(loop_length + extra)

    for i in range(steps):
        start = i * step_samples

        if kick_pattern[i] == 1:
            k = kick(duration=step_duration)
            output[start:start+len(k)] += k

        if snare_pattern[i] == 1:
            s = snare_bp(duration=step_duration)
            output[start:start+len(s)] += s

        if hihat_pattern[i] == 1:
            h = hi_hat(duration=step_duration * 0.5)
            output[start:start+len(h)] += h
            
    return output[:loop_length]


    
beat = sequencer(kick_pattern, snare_pattern, hihat_pattern, bpm=87)
stream = loop_audio(beat)
stream.start()

while True:
    pass  
