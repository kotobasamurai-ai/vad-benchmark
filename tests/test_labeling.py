import numpy as np

from vad_benchmark.labeling import num_frames, resample_probs_to_grid, segments_to_labels
from vad_benchmark.types import SpeechSegment


def test_num_frames_exact():
    # 10 seconds -> 10000 / 31.25 = 320 frames
    assert num_frames(10.0) == 320


def test_segments_to_labels_full_speech():
    segs = (SpeechSegment(0.0, 10.0),)
    labels = segments_to_labels(segs, 10.0)
    assert labels.shape == (320,)
    assert labels.sum() == 320


def test_segments_to_labels_empty():
    labels = segments_to_labels((), 5.0)
    assert labels.sum() == 0


def test_segments_to_labels_partial():
    # speech from 1.0 to 2.0 -> frames 32..63 inclusive ish
    segs = (SpeechSegment(1.0, 2.0),)
    labels = segments_to_labels(segs, 5.0)
    assert labels.sum() == 32  # 1 second = 32 frames of 31.25ms


def test_resample_probs_same_hop_is_identity():
    probs = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    out = resample_probs_to_grid(probs, 31.25, 4)
    np.testing.assert_allclose(out, probs)


def test_resample_probs_upsample():
    probs = np.array([0.0, 1.0], dtype=np.float32)
    out = resample_probs_to_grid(probs, 62.5, 4)
    assert out.shape == (4,)
    assert out[0] == 0.0
    assert out[-1] == 1.0
