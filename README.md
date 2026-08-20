# User-trained rhythmic sound detector

This is a small, local Python program for detecting a particular percussive
sound, including a cymbal hit. It does not use a pretrained drum classifier.
Instead, it learns a profile from recordings you provide.

The detector uses two checks:

1. The signal must contain a sudden broadband attack above the recent noise floor.
2. The attack and its first 240 ms must resemble the enrolled examples by
   frequency-specific decay, spectral shape, envelope, and timbre.

The detector is intentionally looking for the start of a hit. A cymbal's
continuing resonance is not treated as a new hit: enrollment fingerprints are
aligned to the first attack, and the resonance tail is automatically added as
rejection data.

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

Record or copy one clean example per file. Use at least 3-5 slightly different
target hits; they make the positive-only profile less brittle. Negative
examples are optional refinements, not a requirement to enumerate every sound
that should be ignored.

```bash
python sound_detector.py enroll \
  examples/hit-1.wav examples/hit-2.wav examples/hit-3.wav \
  --profile hit.json
```

Each file can contain silence around one hit. During enrollment the program
finds the beginning of its strongest broadband attack and saves the attack plus
early decay fingerprint in `hit.json`. A later portion of each example is also
stored as a rejection example so the cymbal's resonance is not mistaken for
another hit. The original recordings are not copied into the profile.

Explicit negatives remain available when one particular sound is troublesome:

```bash
python sound_detector.py enroll examples/hit-1.wav examples/hit-2.wav \
  --negative examples/clap.wav --profile hit.json
```

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

The positive fingerprint is the main protection against false positives.
Useful tuning options:

```bash
python sound_detector.py detect recording.wav --profile hit.json \
  --threshold 0.72 --jump-db 4 --min-dbfs -40 --verbose
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

The listener keeps a short alignment buffer, waits roughly 250 ms after a
candidate onset, then prints a detection after classifying the attack and early
decay. The
first half-second is used to measure the room and produces no detections. The
default 4 dB loudness jump and 0.72 similarity threshold are starting points.
The identity fingerprint—not volume—is what should accept or reject a
candidate. Clips shorter than 160 ms are refused because they discard the
early decay that distinguishes a cymbal from an ordinary impact.

The live detector can analyze overlapping hit windows, so it does not need to
wait for one sound to finish before watching for the next attack. For very fast
or quiet playing, these controls make the onset gate more responsive:

```bash
python sound_detector.py listen --profile hit.json \
  --min-gap-ms 45 --jump-db 3 --rise-db 3 --min-dbfs -68 --verbose
```

`--min-gap-ms` is the shortest allowed time between candidate hits.
`--rise-db` is the short-term increase that marks a new attack. Spectral flux is
also required, which helps reject gradual cymbal resonance. Lower either
value carefully if hits are missed; the fingerprint still decides whether each
candidate is the enrolled sound.

For 200 ms after a candidate hit, the detector applies a stricter 5 dB rise
requirement. This rejects low-rise decay and reverb without blocking a genuinely
sharp following hit. Change it with `--post-hit-rise-db` if necessary.

Profiles created with older versions must be re-enrolled. Version 4 profiles
use corrected attack alignment and retain early decay and timbre information;
rejection data can only lower a candidate's score, never raise it.

## Limitations

- Enrollment assumes one main hit per example recording.
- The profile is intentionally specific to the sound and recording setup; if
  the microphone or room changes substantially, enroll a few new examples.
- `sounddevice` is needed only for `listen`; file enrollment and scanning need
  NumPy and SoundFile.
