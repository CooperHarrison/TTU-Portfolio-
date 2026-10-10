# Synth usage:
# - Install required audio and keyboard packages:
#   py -m pip install numpy sounddevice keyboard
# - For optional MIDI input, also install:
#   py -m pip install mido python-rtmidi
# - Run this script locally with an available stereo audio output device.
# - Hold A W S E D F T G Y H U J K to play notes; - / = shifts the octave down/up, from C2 to C7.
# - Press 1-4 to select sine, square, saw, or triangle waveform.
# - Z/X changes attack, C/V decay, B/N sustain, M and , for release, and [ and ] keys for filter cutoff.
# - Press Ctrl+C to stop the synth.
# - This will have to be run locally, as the online environment does not support audio output or keyboard input.
import time
import threading
from contextlib import nullcontext

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
keyboard_note_keys = {
    "a": 60, "w": 61, "s": 62, "e": 63, "d": 64,
    "f": 65, "t": 66, "g": 67, "y": 68, "h": 69,
    "u": 70, "j": 71, "k": 72,
}
keyboard_held_keys = set()
keyboard_held_notes = set()
midi_held_notes = set()
keyboard_key_notes = {}
keyboard_octave = 4
octave_control_keys_held = set()


def find_output_device():
    if sd is None:
        return None
    devices = sd.query_devices()
    if not devices:
        return None

    default_output = sd.default.device[1]
    if default_output is not None and default_output >= 0:
        device = sd.query_devices(default_output)
        if device["max_output_channels"] >= 2:
            return int(default_output)

    for index, device in enumerate(devices):
        if device["max_output_channels"] >= 2:
            return index
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


def start_note(note, source_notes):
    with notes_lock:
        source_notes.add(note)
        active_notes.add(note)
        note_env[note] = 0.0
        note_phases[note] = 0.0
        new_note_flag[note] = True
        note_state[note] = "attack"
        note_filter[note] = (0.0, 0.0)


def stop_note(note, source_notes, other_source_notes):
    with notes_lock:
        source_notes.discard(note)
        if note not in other_source_notes:
            active_notes.discard(note)
            if note in note_state:
                note_state[note] = "release"


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
print("Synth running... Press Ctrl+C to stop.")
if keyboard is not None:
    print("Keyboard notes: A W S E D F T G Y H U J K (one octave at a time)")
    print("- / = shifts the keyboard octave (starting at C2 through C7)")
    print("Waveform: 1 = sine, 2 = square, 3 = saw, 4 = triangle")
    print("Z/X attack, C/V decay, B/N sustain, M/, release; [ / ] = filter cutoff")
else:
    print("Computer-keyboard input is unavailable. Install the 'keyboard' package to enable it.")

midi_port_name = find_midi_input(midi_name)
if mido is None:
    print("mido is not installed. MIDI input is unavailable; continuing without MIDI.")
elif midi_port_name is None:
    print("No MIDI input device found. Continuing with computer-keyboard input.")

if keyboard is None and midi_port_name is None:
    print("No computer keyboard or MIDI input is available. Install 'keyboard' or connect a MIDI device.")
    stream.stop()
    stream.close()
    raise SystemExit(1)

try:
    port_context = mido.open_input(midi_port_name) if midi_port_name is not None else nullcontext()
    with port_context as port:
        while True:
            if keyboard is not None and hasattr(keyboard, 'is_pressed'):
                if keyboard.is_pressed('1'):
                    current_waveform = "sine"
                elif keyboard.is_pressed('2'):
                    current_waveform = "square"
                elif keyboard.is_pressed('3'):
                    current_waveform = "saw"
                elif keyboard.is_pressed('4'):
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

                if keyboard.is_pressed(']'):
                    cutoff = min(20000.0, cutoff + 10)
                if keyboard.is_pressed('['):
                    cutoff = max(50.0, cutoff - 10)

                for key, octave_delta in (("-", -1), ("=", 1)):
                    pressed = keyboard.is_pressed(key)
                    if pressed and key not in octave_control_keys_held:
                        octave_control_keys_held.add(key)
                        keyboard_octave = max(2, min(7, keyboard_octave + octave_delta))
                        print(f"Keyboard octave: C{keyboard_octave}")
                    elif not pressed:
                        octave_control_keys_held.discard(key)

                for key, note in keyboard_note_keys.items():
                    pressed = keyboard.is_pressed(key)
                    if pressed and key not in keyboard_held_keys:
                        keyboard_held_keys.add(key)
                        note += 12 * (keyboard_octave - 4)
                        keyboard_key_notes[key] = note
                        start_note(note, keyboard_held_notes)
                    elif not pressed and key in keyboard_held_keys:
                        keyboard_held_keys.remove(key)
                        note = keyboard_key_notes.pop(key)
                        stop_note(note, keyboard_held_notes, midi_held_notes)

                a = 1.0 - np.exp(-2.0 * np.pi * cutoff / fs)

            attack_inc = 1.0 / (env_attack * fs)
            decay_inc = (1.0 - env_sustain) / (env_decay * fs)
            release_inc = 1.0 / (env_release * fs)

            if port is not None:
                for msg in port.iter_pending():
                    if msg.type == "note_on" and msg.velocity > 0:
                        start_note(msg.note, midi_held_notes)
                    elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                        stop_note(msg.note, midi_held_notes, keyboard_held_notes)

            time.sleep(0.002)
except KeyboardInterrupt:
    pass
finally:
    if 'stream' in locals() and stream is not None:
        stream.stop()
        stream.close()