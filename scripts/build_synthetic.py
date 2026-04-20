"""Build a small synthetic speech+noise dataset for precision/recall testing.

- Takes ESC-50 noise clips (already downloaded, ~5s each, all non-speech).
- Takes LibriSpeech dev-clean style speech clips from any user-provided folder, OR
  alternately downloads a tiny subset of common-voice Japanese via soundfile
  (skipped here to stay offline-friendly).

For the first iteration we assemble:
    [silence 1s] [speech clip] [silence 1s] [noise clip] [silence 1s] ...

and emit labels as RTTM-ish JSON so the existing synthetic loader can read them.

Usage:
    python scripts/build_synthetic.py \
        --speech-dir data/librispeech/dev-clean \
        --noise-dir data/esc50/audio \
        --out data/synthetic \
        --num-clips 40
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
SIL = 1.0  # seconds of silence between clips


def _load_resampled(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import resampy

        wav = resampy.resample(wav, sr, SR).astype(np.float32)
    return wav.astype(np.float32)


def build(
    speech_dir: Path,
    noise_dir: Path,
    out_dir: Path,
    num_clips: int,
    seed: int = 0,
) -> None:
    rng = random.Random(seed)
    speech_files = sorted(list(speech_dir.rglob("*.flac")) + list(speech_dir.rglob("*.wav")))
    noise_files = sorted(noise_dir.glob("*.wav"))
    if not speech_files:
        raise SystemExit(f"no speech files under {speech_dir}")
    if not noise_files:
        raise SystemExit(f"no noise files under {noise_dir}")

    rng.shuffle(speech_files)
    rng.shuffle(noise_files)
    speech_files = speech_files[:num_clips]
    noise_files = noise_files[:num_clips]

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    silence = np.zeros(int(SR * SIL), dtype=np.float32)

    items = []
    for idx, (sp, ns) in enumerate(zip(speech_files, noise_files)):
        sp_wav = _load_resampled(sp)
        ns_wav = _load_resampled(ns)

        # random order: speech first or noise first
        if rng.random() < 0.5:
            parts = [("sil", silence), ("sp", sp_wav), ("sil", silence), ("ns", ns_wav)]
        else:
            parts = [("sil", silence), ("ns", ns_wav), ("sil", silence), ("sp", sp_wav)]
        parts.append(("sil", silence))

        segments = []
        buf = []
        t = 0.0
        for kind, arr in parts:
            dur = len(arr) / SR
            if kind == "sp":
                segments.append({"start": t, "end": t + dur})
            buf.append(arr)
            t += dur

        full = np.concatenate(buf)
        name = f"synth_{idx:03d}"
        sf.write(str(audio_dir / f"{name}.wav"), full, SR, subtype="PCM_16")
        items.append({"utt_id": name, "duration": t, "speech_segments": segments})

    (out_dir / "labels.json").write_text(json.dumps(items, indent=2))
    print(f"wrote {len(items)} clips to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--speech-dir", type=Path, required=True)
    p.add_argument("--noise-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-clips", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build(a.speech_dir, a.noise_dir, a.out, a.num_clips, a.seed)


if __name__ == "__main__":
    main()
