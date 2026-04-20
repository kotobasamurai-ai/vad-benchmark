# vad-benchmark

Local reproduction of the [Silero-VAD quality metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics),
extended with ai-coustics' Quail VAD.

## What it does

For each (dataset, engine) pair:

1. Loads an audio file plus a ground-truth speech/non-speech segmentation.
2. Runs the VAD engine and gets a speech probability per 31.25 ms frame
   (the same grid the Silero wiki uses).
3. Computes ROC-AUC and per-frame Accuracy against the ground truth.

Engines:

- `webrtc` — Google WebRTC VAD (via `webrtcvad-wheels`)
- `silero` — [Silero VAD](https://github.com/snakers4/silero-vad) v5 or v6
- `aicoustics` — [ai-coustics Quail VAD](https://docs.ai-coustics.com/guides/voice-activity-detection.md)
  (requires `AIC_SDK_LICENSE`)

## Datasets

From the Silero wiki's multi-domain validation set (public subset only):

| ID | Duration used | Source |
|---|---|---|
| `esc50` | 2.7 h | https://github.com/karolpiczak/ESC-50 |
| `voxconverse` | 2 h sample | https://github.com/joonson/voxconverse |
| `msdwild` | 2 h sample | https://github.com/X-LANCE/MSDWILD |
| `alimeeting` | 2 h sample | https://www.openslr.org/119/ |
| `aishell4` | 2 h sample | https://www.openslr.org/111/ |
| `earnings21` | 2 h sample | https://github.com/revdotcom/speech-datasets |
| `libriparty` | 2 h sample | https://drive.google.com/file/d/1--cAS5ePojMwNY5fewioXAv9YlYAWzIJ/view |

The two "private" sources from the wiki are not reproducible.

## Quickstart

```bash
# 1. Install (pick the extras you need)
pip install -e '.[webrtc,silero,aicoustics]'

# 2. Get your AI-coustics key
export AIC_SDK_LICENSE=...

# 3. Put audio + labels under data/ (see configs/datasets.yaml for layout)

# 4. Smoke test on a small subset
vad-bench run \
    --dataset esc50 voxconverse \
    --engine webrtc silero aicoustics \
    --max-seconds-per-dataset 600 \
    --out results/smoke.json

# 5. Full run
vad-bench run --all --out results/full.json
vad-bench report results/full.json
```

## Layout

```
src/vad_benchmark/
  engines/     # VadEngine implementations
  datasets/    # dataset loaders -> (audio_path, speech_segments)
  labeling.py  # segment -> per-frame 0/1 on the common 31.25ms grid
  metrics.py   # ROC-AUC, Accuracy
  cli.py
```
