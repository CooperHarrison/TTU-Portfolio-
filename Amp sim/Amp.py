import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, lfilter


sd.default.device = (57, 53)
sr = 44100
blocksize = 128


ir, _ = sf.read(r"C:\Users\pokem\Desktop\Coding\Amp sim\IR1.wav")
if ir.ndim > 1:
    ir = ir[:, 0]

ir = ir / np.max(np.abs(ir))
ir = ir * 0.2
L = len(ir)

N = 1
while N < L + blocksize - 1:
    N *= 2

IR_FFT = np.fft.rfft(ir, n=N).astype(np.complex64)
overlap = np.zeros(N - blocksize, dtype=np.float32)

def process_block_fft(block):
    global overlap

    x = np.zeros(N, dtype=np.float32)
    x[:blocksize] = block

    X = np.fft.rfft(x)
    Y = X * IR_FFT
    y = np.fft.irfft(Y).astype(np.float32)

    y[:len(overlap)] += overlap

    out = y[:blocksize].copy()
    overlap = y[blocksize:]

    return out

def hpf(cutoff):
    b, a = butter(2, cutoff / (0.5 * sr), btype='high')
    return b, a
def lpf(cutoff):
    b, a = butter(2, cutoff / (0.5 * sr), btype='low')
    return b, a


hp1_b, hp1_a = hpf(120)
lp1_b, lp1_a = lpf(8000)
hp2_b, hp2_a = hpf(150)
lp2_b, lp2_a = lpf(7000)
hp3_b, hp3_a = hpf(200)
lp3_b, lp3_a = lpf(6000)
hp4_b, hp4_a = hpf(90)
lp4_b, lp4_a = lpf(6500)
hp5_b, hp5_a = hpf(80)
lp5_b, lp5_a = lpf(6000)
hp1_zi = np.zeros(max(len(hp1_b), len(hp1_a)) - 1)
lp1_zi = np.zeros(max(len(lp1_b), len(lp1_a)) - 1)
hp2_zi = np.zeros(max(len(hp2_b), len(hp2_a)) - 1)
lp2_zi = np.zeros(max(len(lp2_b), len(lp2_a)) - 1)
hp3_zi = np.zeros(max(len(hp3_b), len(hp3_a)) - 1)
lp3_zi = np.zeros(max(len(lp3_b), len(lp3_a)) - 1)
hp4_zi = np.zeros(max(len(hp4_b), len(hp4_a)) - 1)
lp4_zi = np.zeros(max(len(lp4_b), len(lp4_a)) - 1)
hp5_zi = np.zeros(max(len(hp5_b), len(hp5_a)) - 1)
lp5_zi = np.zeros(max(len(lp5_b), len(lp5_a)) - 1)

def preamp_stage1(x, gain=30.0, bias=.2, drive=3.0):
    global hp1_zi, lp1_zi
    y, hp1_zi = lfilter(hp1_b, hp1_a, x, zi=hp1_zi)
    y = y * gain + bias
    y = np.tanh(y * drive)
    y, lp1_zi = lfilter(lp1_b, lp1_a, y, zi=lp1_zi)
    return y

def preamp_stage2(x, gain=30.0, bias=.2, drive=3.0):
    global hp2_zi, lp2_zi
    y, hp2_zi = lfilter(hp2_b, hp2_a, x, zi=hp2_zi)
    y = y * gain + bias
    y = np.tanh(y * drive)
    y, lp2_zi = lfilter(lp2_b, lp2_a, y, zi=lp2_zi)
    return y

def preamp_stage3(x, gain=30.0, bias=.2, drive=3.0):
    global hp3_zi, lp3_zi
    y, hp3_zi = lfilter(hp3_b, hp3_a, x, zi=hp3_zi)
    y = y * gain + bias
    y = np.tanh(y * drive)
    y, lp3_zi = lfilter(lp3_b, lp3_a, y, zi=lp3_zi)
    return y

def preamp_stage4(x, gain=30.0, bias=.2, drive=3.0):
    global hp4_zi, lp4_zi
    y, hp4_zi = lfilter(hp4_b, hp4_a, x, zi=hp4_zi)
    y = y * gain + bias
    y = np.tanh(y * drive)
    y, lp4_zi = lfilter(lp4_b, lp4_a, y, zi=lp4_zi)
    return y
def preamp_stage5(x, gain=30.0, bias=.2, drive=3.0):
    global hp5_zi, lp5_zi
    y, hp5_zi = lfilter(hp5_b, hp5_a, x, zi=hp5_zi)
    y = y * gain + bias
    y = np.tanh(y * drive)
    y, lp5_zi = lfilter(lp5_b, lp5_a, y, zi=lp5_zi)
    return y

def tone_stack(x, bass=0.5, mid=0.5, treble=0.5):
    b_bass, a_bass = butter(1, 200/(sr/2), btype='low')
    low = lfilter(b_bass, a_bass, x)
    x = x * (1-bass) + low * bass

    b_mid, a_mid = butter(1, [400/(sr/2), 1500/(sr/2)], btype='bandstop')
    mid_scooped = lfilter(b_mid, a_mid, x)
    x = x * mid + mid_scooped * (1-mid)

    b_treble, a_treble = butter(1, 3000/(sr/2), btype='high')
    high = lfilter(b_treble, a_treble, x)
    x = x * (1-treble) + high * treble

    return x

def preamp(x):
    x = preamp_stage1(x, gain=1.8, bias=-0.00, drive=1.5)
    x = preamp_stage2(x, gain=2.0, bias=-0.00, drive=2.0)
    x = preamp_stage3(x, gain=3.0, bias=0.00, drive=2.5)
    x = preamp_stage4(x, gain=1.5, bias=0.00, drive=3.0)  
    x = tone_stack(x, bass=0.5, mid=0.5, treble=0.5)
    x = preamp_stage5(x, gain=2.5, bias=0.0, drive=2.0)

    return x

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    x = indata[:, 0].astype(np.float32)
    y = x * 0.7
    y = preamp(y)
    y = y * 0.3
    y = process_block_fft(y)
    y = np.clip(y, -1.0, 1.0)

    outdata[:, 0] = y
    outdata[:, 1] = y
#Run
with sd.Stream(channels=2, callback=callback, samplerate=sr, blocksize=blocksize):
    print("Playing... Ctrl+C to stop.")
    while True:
        sd.sleep(1000)