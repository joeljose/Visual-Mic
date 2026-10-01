"""Check scripts/eval_audio.py recovers a known lag, sign and SNR."""

import os
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import eval_audio


def test_known_lag_sign_and_snr(tmp_path):
    rate = 2200
    rng = np.random.RandomState(0)
    ref = rng.randn(rate * 4)
    lag = 300
    rec = np.concatenate([rng.randn(lag) * 0.1, -0.5 * ref])  # delayed, inverted, scaled
    rec = rec + rng.randn(len(rec)) * 0.05  # 20 dB below the signal
    wavfile.write(tmp_path / 'ref.wav', rate, (ref / 8 * 32767).astype(np.int16))
    wavfile.write(tmp_path / 'rec.wav', rate, (rec / 4 * 32767).astype(np.int16))

    s = eval_audio.evaluate(str(tmp_path / 'rec.wav'), str(tmp_path / 'ref.wav'))

    assert s['lag'] == lag
    assert abs(s['snr_db'] - 20) < 1.5
    assert s['coherence'] > 0.95
