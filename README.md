# User-trained rhythmic sound detector

This is a small, local Python program for detecting a particular loud,
short sound: a drum hit, hand clap, stick click, foot tap, or another rhythmic
transient. It does not use a pretrained drum classifier. Instead, it learns a
profile from recordings you provide.

The detector uses two checks:

1. The signal must be a large volume increase above the recent noise floor.
2. The short sound must resemble the enrolled examples by spectral shape and
   attack/decay envelope.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, `sounddevice` may need PortAudio if it is not already available:

```bash
brew install portaudio
```

## Enroll examples

Record or copy one clean example per file. Slightly different recordings are
recommended; they make the profile less brittle. Since loudness alone is not
enough to identify a sound, also record a few loud sounds that should *not* be
detected and pass them as rejection examples.

```bash
python sound_detector.py enroll examples/hit-1.wav examples/hit-2.wav examples/hit-3.wav \
  --negative examples/clap.wav --negative examples/tap.wav \
  --profile hit.json
```

Each file can contain silence around one hit. During enrollment the program
finds the loudest transient and saves only its compact fingerprint in
`hit.json`. The original recordings are not copied into the profile.

## Calibrate with the actual microphone

An MP3 played through a speaker does not sound identical at the microphone.
Speaker EQ, the room, distance, and microphone response all alter its
fingerprint. Add five examples captured through the real setup:

```bash
python sound_detector.py calibrate --profile hit.json --count 5
```

The command prompts for each hit and saves it immediately. Make only the target
sound while each one-second capture is running. These acoustic examples are
added to the original file examples rather than replacing them.

Then record the normal room as rejection data. Let the usual fans, conversation,
music, or other background sounds continue, but do not make the target hit:

```bash
python sound_detector.py calibrate-background \
  --profile hit.json --seconds 6 --replace
```

The loudest short ambient windows are stored as negative templates. Repeat this
calibration when the room or microphone setup changes.

## Scan an uploaded/recorded file

Pass the path to a WAV, FLAC, OGG, or another format supported by `soundfile`:

```bash
python sound_detector.py detect uploaded_recording.wav --profile hit.json
```

The terminal prints updates such as:

```text
[00:02.41] DETECTED score=0.91 level=-12.7 dBFS
```

The `--negative` examples are the most important protection against false
positives. They should include the loud sounds that commonly happen in your
real setup. Useful tuning options:

```bash
python sound_detector.py detect recording.wav --profile hit.json \
  --threshold 0.65 --jump-db 4 --min-dbfs -40 --verbose
```

Raise `--threshold` to reject more borderline sounds; lower it only if valid
variations are being rejected. Lower `--jump-db` if hits are not much louder
than the room. Lower `--min-dbfs` if the microphone signal is quiet.
`--verbose` shows loud events that failed the enrolled-sound match.

## Listen to the microphone

```bash
python sound_detector.py listen --profile hit.json
```

The listener opens a live waveform window showing the last two seconds heard by
the microphone. Its diagnostics bar displays the current RMS level, peak level,
estimated noise floor, trigger gate, and fixed waveform full-scale level. The
default display is fixed at 0 dBFS, which shows the complete `hit.mp3` waveform
without clipping and never changes as room volume changes.

The window also has a **Minimum sound level** slider. This is an absolute
silence gate: candidates below it are ignored before fingerprint matching. Set
it roughly 3–6 dB above the level shown during silence, while keeping it below
the level of your quietest intended hit. The default is −40 dBFS.

For a permanently zoomed-in view of a quieter microphone, choose another fixed
full-scale value. Samples louder than that value clip only in the visualization;
detection is unaffected:

```bash
python sound_detector.py listen --profile hit.json --display-dbfs -12
```

Closing the window stops listening. Use terminal-only mode when needed:

```bash
python sound_detector.py listen --profile hit.json --no-window
```

Press Ctrl-C to stop terminal-only mode. To choose an input device, first list
devices with:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Then pass its numeric ID or name:

```bash
python sound_detector.py listen --profile hit.json --device 2
```

The listener keeps a short alignment buffer, waits about 90 ms after a loud
onset, then prints a detection after classifying the complete short clip. The
first half-second is used to measure the room and produces no detections. The
default 4 dB loudness jump and 0.60 similarity threshold are starting points;
rejection examples and a higher threshold are better than relying on volume
alone. To trade some accuracy for a faster response, enroll with a shorter clip:

```bash
python sound_detector.py enroll hit.mp3 --clip-seconds 0.055 --pre-seconds 0.006 \
  --profile hit-fast.json
python sound_detector.py listen --profile hit-fast.json --block-ms 6
```

The live detector can analyze overlapping hit windows, so it does not need to
wait for one sound to finish before watching for the next attack. For very fast
or quiet playing, these controls make the onset gate more responsive:

```bash
python sound_detector.py listen --profile hit.json \
  --min-gap-ms 45 --jump-db 3 --rise-db 3 --min-dbfs -68 --verbose
```

`--min-gap-ms` is the shortest allowed time between candidate hits.
`--rise-db` is the short-term increase that marks a new attack. Lower either
value carefully if hits are missed; the fingerprint still decides whether each
candidate is the enrolled sound. Use these softer settings only after recording
background rejection examples.

For 200 ms after a candidate hit, the detector applies a stricter 5 dB rise
requirement. This rejects low-rise decay and reverb without blocking a genuinely
sharp following hit. Change it with `--post-hit-rise-db` if necessary.

Profiles created with older versions should be re-enrolled because the new
profile preserves time-varying spectral detail.

## Limitations

- Enrollment assumes one main hit per example recording.
- The profile is intentionally specific to the sound and recording setup; if
  the microphone or room changes substantially, enroll a few new examples.
- `sounddevice` is needed only for `listen`; file enrollment and scanning need
  NumPy and SoundFile.
