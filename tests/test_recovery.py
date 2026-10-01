"""Recovery tests against a known motion signal (CPU path).

A blurred random texture is moved with exact sub-pixel shifts (Fourier
shift theorem) following a 100-1000 Hz linear chirp, the test signal of
Davis et al. 2014, Fig. 5. Motion is given in RMS pixels; the paper measured
0.002-0.029 px on real objects (Fig. 8b). Frames get Gaussian sensor noise
and uint8 rounding, and are streamed through extract_audio one at a time.

The injected motion is the ground truth. Real recovery also passes through
the object's frequency response, which these tests don't model.

Score: scale- and sign-invariant SNR (dB) of the output against the truth,
best lag within +/-20 samples, both high-passed at 100 Hz unless stated.

Tests marked xfail document known failures; strict=True makes them fail
loudly once the referenced issue is fixed, so the marker gets removed.
"""

import contextlib
import functools
import io
import os
import sys

import numpy as np
import pytest
from scipy import ndimage, signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import visualmic

pytest.importorskip("dtcwt")

FPS = 2200
N = 1100  # 0.5 s
SIZE = 128
T = np.arange(N) / FPS
TRUTH = signal.chirp(T, 100, T[-1], 1000)
TRUTH /= np.sqrt(np.mean(TRUTH ** 2))  # unit RMS, so amplitude is RMS px

_tex = ndimage.gaussian_filter(np.random.RandomState(0).rand(SIZE, SIZE), 2.0)
_tex = 40 + 170 * (_tex - _tex.min()) / (_tex.max() - _tex.min())
TEX_F = np.fft.fft2(_tex)
KY = np.fft.fftfreq(SIZE)[:, None]
KX = np.fft.fftfreq(SIZE)[None, :]


class SyntheticCap:
    """cv2.VideoCapture stand-in that renders frames on demand."""

    def __init__(self, rms_px, angle=0.0, drift_px=0.0, noise=1.0, seed=1):
        self.d = rms_px * TRUTH + drift_px * T / T[-1]
        self.cos, self.sin = np.cos(angle), np.sin(angle)
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        self.i = 0

    def read(self):
        if self.i >= N:
            return False, None
        d = self.d[self.i]
        self.i += 1
        phase = np.exp(-2j * np.pi * (KX * d * self.cos + KY * d * self.sin))
        img = np.real(np.fft.ifft2(TEX_F * phase)) + self.rng.randn(SIZE, SIZE) * self.noise
        gray = np.clip(np.round(img), 0, 255).astype(np.uint8)
        return True, np.repeat(gray[:, :, None], 3, axis=2)

    def release(self):
        pass


@functools.lru_cache(maxsize=None)
def recover(rms_px, angle=0.0, drift_px=0.0, freq_low=100, reported=N):
    with contextlib.redirect_stdout(io.StringIO()):
        return visualmic.extract_audio(
            SyntheticCap(rms_px, angle, drift_px), reported, 3, 6, 0, FPS, freq_low, None)


def snr_db(out, highpass=100):
    ref = TRUTH
    if highpass:
        sos = signal.butter(4, highpass / (FPS / 2), 'high', output='sos')
        out, ref = signal.sosfiltfilt(sos, out), signal.sosfiltfilt(sos, ref)
    y = ref[50:-50]
    best = -np.inf
    for lag in range(-20, 21):
        x = np.roll(out, lag)[50:-50]
        fit = (x @ y) / (y @ y) * y
        best = max(best, 10 * np.log10((fit @ fit) / ((x - fit) @ (x - fit))))
    return best


# v2.0.0 (band alignment by cross-correlation to a fixed reference band):
# horizontal 0.005 px -22 dB, 0.01 px -19 dB, vertical 0.01 px 1.5 dB,
# diagonal 0.01 px 12 dB. Now (bands combined into x/y motion, projected on
# the dominant direction): 9.6, 15.8, 16.1 and 15.9 dB.
@pytest.mark.parametrize("rms_px,angle", [
    pytest.param(0.005, 0.0, id="horizontal-0.005px"),
    pytest.param(0.01, 0.0, id="horizontal-0.01px"),
    pytest.param(0.01, np.pi / 2, id="vertical-0.01px"),
    pytest.param(0.01, np.pi / 4, id="diagonal-0.01px"),
    pytest.param(0.01, 3 * np.pi / 4, id="antidiagonal-0.01px"),
])
def test_recovers_paper_scale_motion(rms_px, angle):
    assert snr_db(recover(rms_px, angle)) >= 8


# Davis et al. Eq. 8: SNR grows linearly with motion amplitude, so doubling
# it adds about 6 dB (+6.2 dB now; v2.0.0 showed no such trend).
def test_snr_follows_motion_amplitude():
    gain = snr_db(recover(0.01)) - snr_db(recover(0.005))
    assert 4 <= gain <= 8


# 0.3 px of drift over the clip with the CLI's default high-pass. Scored
# without a high-pass, since that is what the user hears. v2.0.0 (no filter
# by default) gave -18.5 dB.
def test_default_settings_reject_drift():
    assert snr_db(recover(0.01, drift_px=0.3, freq_low=visualmic.default_freq_low(FPS)), highpass=None) >= 10


# 3 px of drift wraps the phase measured against frame 0. Still about 3 dB
# with the alignment fix; needs frame-to-frame phase differences.
@pytest.mark.xfail(strict=True, reason="#17: phase wraps relative to frame 0")
def test_large_drift_does_not_wrap():
    assert snr_db(recover(0.01, drift_px=3.0)) >= 10


# Containers often report the wrong frame count (#16). Every decodable frame
# should be used, whatever the metadata says.
@pytest.mark.parametrize("reported", [N // 2, N + 300], ids=["under-reported", "over-reported"])
def test_output_length_follows_decoded_frames(reported):
    assert len(recover(0.01, reported=reported)) == N
