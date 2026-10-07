#This doesn't work in a virtual enviroment like here on github, It would need to be run locally, but this is the code.

import sys
import time
import threading

import numpy as np

try:
    import sounddevice as sd
except ModuleNotFoundError:
    sd = None

try:
    import keyboard
except ModuleNotFoundError:
    keyboard = None

try:
    import mido
except ModuleNotFoundError:
    mido = None


fs = 44100
volume = 0.12


env_attack = 0.005
env_release = 0.02
env_decay = 0.05
env_sustain = 0.7

attack_inc = 1.0 / (env_attack * fs)
decay_inc = (1.0 - env_sustain) / (env_decay * fs)
release_inc = 1.0 / (env_release * fs)


cutoff = 2000.0
a = 1.0 - np.exp(-2.0 * np.pi * cutoff / fs)
resonance = 0.5



notes_lock = threading.Lock()
active_notes = set()

note_phases = {}
note_env = {}
note_state = {}
note_filter = {}
new_note_flag = {}

current_waveform = "sine"

midi_name = "Roland Digital Piano 0"

if sd is not None:
    try:
        if sys.platform.startswith("win"):
            sd.default.device = 18
            sd.default.extra_settings = sd.WasapiSettings(exclusive=True)
    except Exception:
        pass


def find_output_device():
    if sd is None:
        return None
    try:
        devices = sd.query_devices()
        if not devices:
            return None
        for index, device in enumerate(devices):
            name = str(device.get("name", "")).lower()
            if any(token in name for token in ["default", "speaker", "output", "headphones"]):
                return index
        return None
    except Exception:
        return None


def find_midi_input(target_name=None):
    if mido is None:
        return None
    try:
        ports = mido.get_input_names()
    except Exception:
        return None

    if not ports:
        return None

    if target_name:
        for port_name in ports:
            if target_name.lower() in port_name.lower():
                return port_name

    return ports[0]


def poly_blep(t, dt):
    out = np.zeros_like(t)
    x = (t / (2.0 * np.pi)) % 1.0

    idx = x < dt
    if np.any(idx):
        xx = x[idx] / dt
        out[idx] = xx + xx - xx * xx - 1.0

    idx = x > 1.0 - dt
    if np.any(idx):
        xx = (x[idx] - 1.0) / dt
        out[idx] = xx * xx + xx + xx + 1.0

    return out

def osc_saw_polyblep(t, phase_inc):
    dt = phase_inc / (2.0 * np.pi)
    saw = (t / np.pi) - 1.0
    saw -= np.floor((saw + 1.0) * 0.5) * 2.0
    return saw - poly_blep(t, dt)

def osc_square_polyblep(t, phase_inc):
    dt = phase_inc / (2.0 * np.pi)
    sq = np.sign(np.sin(t))
    sq -= poly_blep(t, dt)
    sq += poly_blep((t + np.pi) % (2.0 * np.pi), dt)
    return sq


def audio_callback(outdata, frames, time_info, status):
    global current_waveform, active_notes, note_phases, note_env
    global new_note_flag, note_state, note_filter, a

    audio = np.zeros(frames, dtype=np.float32)
    dead_voices = []

    with notes_lock:
        notes = set(active_notes)
        all_notes = set(notes) | set(note_env.keys())

    if not all_notes:
        outdata[:] = 0.0
        return

    n = np.arange(frames, dtype=np.float32)

    for note in list(all_notes):


        freq = 440.0 * (2 ** ((note - 69) / 12))
        phase = note_phases.get(note, 0.0)
        phase_inc = (2.0 * np.pi * freq) / fs

        is_new = new_note_flag.get(note, False)
        if is_new:
            phase = 0.0
            note_phases[note] = 0.0
            note_env[note] = 0.0

        t = phase + phase_inc * n
        phase = (phase + phase_inc * frames) % (2.0 * np.pi)
        note_phases[note] = phase

        t_wrapped = np.mod(t, 2.0 * np.pi)

        if current_waveform == "sine":
            wave = np.sin(t_wrapped)
        elif current_waveform == "saw":
            wave = osc_saw_polyblep(t_wrapped, phase_inc)
        elif current_waveform == "square":
            wave = osc_square_polyblep(t_wrapped, phase_inc)
        elif current_waveform == "triangle":
            wave = np.arcsin(np.sin(t_wrapped)) * (2.0 / np.pi)
        else:
            wave = np.sin(t_wrapped)

        if is_new:
            fade_len = min(64, frames)
            fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            wave[:fade_len] *= fade
            new_note_flag[note] = False


        env = note_env.get(note, 0.0)
        state = note_state.get(note, "release")
        env_curve = np.zeros(frames, dtype=np.float32)

        for i in range(frames):
            if state == "attack":
                env += attack_inc
                if env >= 1.0:
                    env = 1.0
                    state = "decay"

            elif state == "decay":
                env -= decay_inc
                if env <= env_sustain:
                    env = env_sustain
                    state = "sustain"

            elif state == "sustain":
                env = env_sustain

            elif state == "release":
                env -= release_inc
                if env <= 0.0:
                    env = 0.0

            env_curve[i] = env

        note_env[note] = env
        note_state[note] = state


        if env == 0.0:
            wave[:] = 0.0


        if env <= 0.0 and state == "release":
            last_y = note_filter.get(note, (0.0, 0.0))
            if isinstance(last_y, tuple):
                last_y = last_y[0]

            tail = min(64, len(audio))
            fade = np.linspace(1.0, 0.0, tail, dtype=np.float32)

            audio[:tail] = audio[:tail] + last_y * fade

            dead_voices.append(note)
            continue

        voice = wave * env_curve
        y1, y2 = note_filter.get(note, (0.0, 0.0))
        filtered = np.zeros(frames, dtype=np.float32)

        for i in range(frames):

            y = y1 + a * (voice[i] - y1)


            y += resonance * (y1 - y2)

            y2 = y1
            y1 = y

            filtered[i] = y

        note_filter[note] = (y1, y2)
        audio += filtered
       
            


    with notes_lock:
        for note in dead_voices:
            note_env.pop(note, None)
            note_phases.pop(note, None)
            new_note_flag.pop(note, None)
            note_state.pop(note, None)
            note_filter.pop(note, None)
            active_notes.discard(note)

    out = (audio * volume).astype(np.float32)
    outdata[:] = np.column_stack((out, out))


device_index = find_output_device()
if sd is None:
    print("sounddevice is not installed. Please install it with: pip install sounddevice")
    raise SystemExit(1)

if device_index is None:
    print("No audio output device detected. The synth cannot start in this environment.")
    raise SystemExit(0)

stream = sd.OutputStream(
    samplerate=fs,
    channels=2,
    callback=audio_callback,
    blocksize=128,
    latency='low',
    dtype='float32',
    device=device_index
)

stream.start()
print("MIDI synth running... Press Ctrl+C to stop.")
print("Q = sine, W = square, E = saw, R = triangle")

midi_port_name = find_midi_input(midi_name)
if mido is None:
    print("mido is not installed. MIDI input is unavailable.")
    stream.stop()
    stream.close()
    raise SystemExit(1)

if midi_port_name is None:
    print(f"No MIDI device matched '{midi_name}'. Waiting for a MIDI input port to appear...")
    midi_port_name = find_midi_input()

if midi_port_name is None:
    print("No MIDI input devices found. Exiting.")
    stream.stop()
    stream.close()
    raise SystemExit(0)

try:
    with mido.open_input(midi_port_name) as port:
        while True:
            if keyboard is not None and hasattr(keyboard, 'is_pressed'):
                if keyboard.is_pressed('q'):
                    current_waveform = "sine"
                elif keyboard.is_pressed('w'):
                    current_waveform = "square"
                elif keyboard.is_pressed('e'):
                    current_waveform = "saw"
                elif keyboard.is_pressed('r'):
                    current_waveform = "triangle"

                if keyboard.is_pressed('z'):
                    env_attack = max(0.0001, env_attack - 0.001)
                if keyboard.is_pressed('x'):
                    env_attack = min(1.0, env_attack + 0.001)

                if keyboard.is_pressed('c'):
                    env_decay = max(0.001, env_decay - 0.001)
                if keyboard.is_pressed('v'):
                    env_decay = min(1.0, env_decay + 0.001)

                if keyboard.is_pressed('b'):
                    env_sustain = max(0.0, env_sustain - 0.01)
                if keyboard.is_pressed('n'):
                    env_sustain = min(1.0, env_sustain + 0.01)

                if keyboard.is_pressed('m'):
                    env_release = max(0.01, env_release - 0.001)
                if keyboard.is_pressed(','):
                    env_release = min(1.0, env_release + 0.001)

                if keyboard.is_pressed('u'):
                    cutoff = min(20000.0, cutoff + 10)
                if keyboard.is_pressed('j'):
                    cutoff = max(50.0, cutoff - 10)

                a = 1.0 - np.exp(-2.0 * np.pi * cutoff / fs)

            attack_inc = 1.0 / (env_attack * fs)
            decay_inc = (1.0 - env_sustain) / (env_decay * fs)
            release_inc = 1.0 / (env_release * fs)

            for msg in port.iter_pending():
                with notes_lock:
                    if msg.type == "note_on" and msg.velocity > 0:
                        active_notes.add(msg.note)
                        note_env[msg.note] = 0.0
                        note_phases[msg.note] = 0.0
                        new_note_flag[msg.note] = True
                        note_state[msg.note] = "attack"
                        note_filter[msg.note] = (0.0, 0.0)

                    elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                        active_notes.discard(msg.note)
                        if msg.note in note_state:
                            note_state[msg.note] = "release"

            time.sleep(0.002)
except KeyboardInterrupt:
    pass
finally:
    if 'stream' in locals() and stream is not None:
        stream.stop()
        stream.close()