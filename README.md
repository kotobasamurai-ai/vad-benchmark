# vad-benchmark

Local reproduction of the [Silero-VAD quality metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics),
extended with [ai-coustics Quail VAD](https://docs.ai-coustics.com/guides/voice-activity-detection.md)
and WebRTC VAD.

For each (dataset, engine) pair we compute per-frame speech probabilities on a 31.25 ms
grid (the same grid Silero's wiki uses) and report **ROC-AUC, accuracy, precision,
recall, F1, FPR**. For binary-output engines (WebRTC, AI-coustics) a parameter sweep
gives a trapezoidal AUC.

## 5-minute quickstart

Runs WebRTC + Silero + AI-coustics on ESC-50 (non-speech, 60 s) and a small synthetic
speech+noise set. Finishes in ~3 min on CPU.

```bash
# 1. Clone and install
git clone https://github.com/kotobasamurai-ai/vad-benchmark.git
cd vad-benchmark
python3 -m venv .venv
.venv/bin/pip install -e '.[webrtc,silero,aicoustics,dev]'

# 2. AI-coustics license (skip if you only want WebRTC + Silero)
echo "AICOUSTICS_LICENSE_KEY=<your key from https://developers.ai-coustics.com>" > .env

# 3. Tiny data + synthetic speech+noise mix
bash scripts/download_esc50.sh
bash scripts/download_librispeech_devclean.sh
.venv/bin/python scripts/build_synthetic.py \
    --speech-dir data/librispeech/LibriSpeech/dev-clean \
    --noise-dir  data/esc50/audio \
    --out        data/synthetic \
    --num-clips  40

# 4. Frame-level metrics on synthetic + esc50
.venv/bin/vad-bench run \
    --dataset synthetic esc50 \
    --engine  webrtc silero aicoustics \
    --max-seconds-per-dataset 180 \
    --out results/quickstart.json

# 5. Sensitivity sweep for binary-output engines
.venv/bin/vad-bench sweep aicoustics webrtc \
    --dataset synthetic esc50 \
    --max-seconds-per-dataset 120 \
    --out results/sweep.json
```

The `run` command prints a Rich table. `sweep` prints one ROC-style table per
(engine, dataset) plus a trapezoidal AUC over the swept operating points.

## Sample output

`vad-bench run` on the quickstart data (3 engines, synthetic + esc50):

| dataset | engine | clips | sec | speech% | ROC-AUC | acc | precision | recall | F1 | FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic | webrtc | 10 | 166 | 51.8 | 0.669 | 0.679 | 0.629 | 0.928 | 0.749 | 0.589 |
| synthetic | silero | 10 | 166 | 51.8 | **0.961** | 0.882 | **0.999** | 0.773 | 0.872 | **0.001** |
| synthetic | aicoustics | 10 | 166 | 51.8 | 0.891 | **0.888** | 0.975 | **0.805** | **0.882** | 0.022 |
| esc50 (all non-speech) | webrtc | 36 | 180 | 0.0 | - | 0.303 | - | - | - | 0.697 |
| esc50 | silero | 36 | 180 | 0.0 | - | **1.000** | - | - | - | **0.000** |
| esc50 | aicoustics | 36 | 180 | 0.0 | - | 0.979 | - | - | - | 0.021 |

**Silero** is the most conservative (FPR ≈ 0, precision ≈ 1) and also the fastest
(~30× faster than AI-coustics on CPU in our measurement). **AI-coustics** catches more
speech (higher recall) at the cost of more false positives. **WebRTC** is unusable on
noisy data.

## Engines

| engine | output | notes |
|---|---|---|
| `webrtc` | binary | `webrtcvad-wheels`. Aggressiveness 0–3. |
| `silero` | continuous prob | `silero-vad` v6, PyTorch (optional ONNX). |
| `aicoustics` | binary | `aic-sdk` ≥ 2.0, Quail VAD (`quail-l-16khz` by default). Needs `AICOUSTICS_LICENSE_KEY`. |

### AI-coustics model choice

The SDK ships several Quail variants. The VAD is "built into" the enhancement model,
so the choice matters. Measured on our synthetic set (166 s, 52 % speech, default
`sensitivity=6`):

| model | precision | recall | F1 | FPR |
|---|---|---|---|---|
| **`quail-l-16khz`** (default) | **0.975** | 0.805 | 0.882 | **0.022** |
| `quail-vf-2.0-l-16khz` (Voice Focus 2.0) | 0.922 | 0.837 | 0.877 | 0.076 |

`quail-l` is the more conservative pick and wins on precision / FPR. `vf-2.0` trades
precision for recall. Override via the `AiCousticsEngine(model_id=...)` constructor
in `src/vad_benchmark/engines/aicoustics.py`.

Binary-output engines are compared fairly via `vad-bench sweep`, which runs each
operating point separately and draws the ROC curve from the resulting `(FPR, TPR)`
points.

## Datasets

Public subset of the Silero wiki multi-domain validation set:

| id | source | auto-download |
|---|---|---|
| `esc50` | https://github.com/karolpiczak/ESC-50 | `scripts/download_esc50.sh` |
| `voxconverse` | https://github.com/joonson/voxconverse | labels auto; audio manual |
| `synthetic` | LibriSpeech dev-clean + ESC-50 | `scripts/build_synthetic.py` |

The two "private" sources in the wiki (private speech / private noise) are not
reproducible.

Adding a new dataset = a loader that yields `AudioItem(audio_path, speech_segments)`
in seconds. See `src/vad_benchmark/datasets/*.py`.

## Layout

```
src/vad_benchmark/
  engines/      # VadEngine implementations (one per engine)
  datasets/     # dataset loaders -> (audio_path, speech_segments)
  labeling.py   # segment -> per-frame 0/1 on the 31.25 ms grid
  metrics.py    # ROC-AUC, precision, recall, F1, FPR
  sweep.py      # trapezoidal AUC from (FPR, TPR) points
  cli.py        # `vad-bench run|sweep|report`
scripts/        # download + synthetic-build helpers
configs/datasets.yaml
tests/          # pytest
```

Tests: `.venv/bin/pytest tests/`.

## CLI reference

```
vad-bench run    --dataset ID ... --engine NAME ... [--max-seconds-per-dataset N]
vad-bench sweep  ENGINE[:v1,v2,...] ... [--dataset ID ...]
vad-bench report RESULTS.json
```

Default sweep values: `webrtc:0,1,2,3`, `aicoustics:2,4,6,8,10,12`. Override with
`aicoustics:1,3,5,7,9,11,13`.

## License

MIT.
