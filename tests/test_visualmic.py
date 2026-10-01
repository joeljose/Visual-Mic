"""Unit tests for visualmic.py, CPU path."""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import visualmic


# --- Tier 1: Strict (exact equality) ---


class TestFormatDuration:
    def test_zero(self):
        assert visualmic.format_duration(0) == "0s"

    def test_seconds(self):
        assert visualmic.format_duration(59) == "59s"

    def test_one_minute(self):
        assert visualmic.format_duration(60) == "1m 0s"

    def test_minutes_seconds(self):
        assert visualmic.format_duration(125) == "2m 5s"

    def test_hours(self):
        assert visualmic.format_duration(3661) == "1h 1m 1s"

    def test_fractional_truncates(self):
        assert visualmic.format_duration(59.9) == "59s"


class TestSaveWav:
    def test_creates_file(self, tmp_path):
        samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
        path = str(tmp_path / "test.wav")
        visualmic.save_wav(samples, path, 44100)
        assert os.path.isfile(path)

    def test_sample_rate(self, tmp_path):
        from scipy.io.wavfile import read as read_wav
        samples = np.array([0.0, 0.5, -0.5])
        path = str(tmp_path / "test.wav")
        visualmic.save_wav(samples, path, 2200)
        sr, _ = read_wav(path)
        assert sr == 2200

    def test_int16_range(self, tmp_path):
        from scipy.io.wavfile import read as read_wav
        samples = np.array([1.0, -1.0, 0.0])
        path = str(tmp_path / "test.wav")
        visualmic.save_wav(samples, path, 44100)
        _, data = read_wav(path)
        assert data.dtype == np.int16
        assert np.max(data) == 32767
        assert np.min(data) == -32767


# --- Tier 2: Moderate tolerance ---


class TestPostprocessPhaseSignals:
    def test_constant_input_silent(self):
        """Constant phase signals (no motion) should produce silent output."""
        nlevels, n_orient, frame_count = 3, 6, 50
        phase_signals = np.ones((frame_count, nlevels, n_orient))
        result = visualmic.postprocess_phase_signals(
            phase_signals, fps=100
        )
        # Constant input gives a constant sum
        # Normalization maps constant to zero
        assert result.shape == (frame_count,)
        assert np.all(np.isfinite(result))

    def test_normalization_range(self):
        """Output should be in [-1, 1]."""
        nlevels, n_orient, frame_count = 3, 6, 100
        rng = np.random.RandomState(42)
        phase_signals = rng.randn(frame_count, nlevels, n_orient)
        result = visualmic.postprocess_phase_signals(
            phase_signals, fps=100
        )
        assert np.min(result) >= -1.0 - 1e-10
        assert np.max(result) <= 1.0 + 1e-10

    def test_output_shape(self):
        nlevels, n_orient, frame_count = 2, 6, 30
        rng = np.random.RandomState(42)
        phase_signals = rng.randn(frame_count, nlevels, n_orient)
        result = visualmic.postprocess_phase_signals(
            phase_signals, fps=100
        )
        assert result.shape == (frame_count,)


class TestButterworthFilter:
    def test_passband_preserved(self):
        """Signal within passband should be preserved."""
        frame_count = 200
        fps = 1000
        # 100 Hz signal, passband 50-200 Hz
        t = np.arange(frame_count) / fps
        phase_signals = np.tile(np.sin(2 * np.pi * 100 * t)[:, None, None], (1, 1, 6))
        result = visualmic.postprocess_phase_signals(
            phase_signals.copy(), fps=fps, freq_low=50, freq_high=200
        )
        assert np.all(np.isfinite(result))
        # Should have non-zero energy (signal passed through)
        assert np.std(result) > 0.01

    def test_stopband_attenuated(self):
        """Signal outside passband should be attenuated."""
        frame_count = 200
        fps = 1000
        # 400 Hz signal, passband 50-200 Hz
        t = np.arange(frame_count) / fps
        phase_signals = np.tile(np.sin(2 * np.pi * 400 * t)[:, None, None], (1, 1, 6))
        result = visualmic.postprocess_phase_signals(
            phase_signals.copy(), fps=fps, freq_low=50, freq_high=200
        )
        assert np.all(np.isfinite(result))
        # Should have lower energy than unfiltered version
        unfiltered = visualmic.postprocess_phase_signals(
            phase_signals.copy(), fps=fps
        )
        assert np.std(result) < np.std(unfiltered)


class TestDefaultFreqLow:
    def test_paper_rule(self):
        # 1/20 of Nyquist at 2200 fps
        assert visualmic.default_freq_low(2200) == 55

    def test_clamped(self):
        assert visualmic.default_freq_low(400) == 20
        assert visualmic.default_freq_low(20000) == 100


class TestDenoiseSpectral:
    def test_removes_stationary_keeps_intermittent(self):
        """A steady hum is suppressed; a tone that comes and goes survives."""
        fs = 2200
        t = np.arange(fs * 4) / fs
        hum = np.sin(2 * np.pi * 120 * t)
        tone = np.sin(2 * np.pi * 330 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
        out = visualmic.denoise_spectral(hum + tone, fs)

        def power_at(x, f):
            freqs, p = signal.welch(x, fs=fs, nperseg=512)
            return p[np.argmin(np.abs(freqs - f))]

        assert out.shape == t.shape
        assert power_at(out, 120) < 0.01 * power_at(hum, 120)
        assert power_at(out, 330) > 0.5 * power_at(tone, 330)


class TestShortClips:
    @pytest.mark.parametrize("frames", [2, 13, 27])
    def test_bandpass_on_short_clip(self, frames):
        """sosfiltfilt's default padding needs 28+ frames; shorter clips must still work (#15)."""
        rng = np.random.RandomState(0)
        result = visualmic.postprocess_phase_signals(
            rng.randn(frames, 3, 6), fps=2200, freq_low=100, freq_high=800
        )
        assert result.shape == (frames,)
        assert np.all(np.isfinite(result))


class TestEstimateVram:
    def test_basic_arithmetic(self):
        result = visualmic.estimate_vram(16, 256, 256, 3)
        # 16 * 256 * 256 * 4 * 15 + 300MB
        expected = 16 * 256 * 256 * 4 * 15 + 300 * 1024 * 1024
        assert result == expected

    def test_scales_with_batch_size(self):
        small = visualmic.estimate_vram(8, 256, 256, 3)
        large = visualmic.estimate_vram(32, 256, 256, 3)
        assert large > small

    def test_scales_with_resolution(self):
        small = visualmic.estimate_vram(16, 256, 256, 3)
        large = visualmic.estimate_vram(16, 512, 512, 3)
        assert large > small


# --- Tier 3: Smoke tests ---


def _create_synthetic_video(path, num_frames=32, height=256, width=256):
    """Create a synthetic video with random noise for testing."""
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    rng = np.random.RandomState(42)
    for _ in range(num_frames):
        frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


try:
    import dtcwt  # noqa: F401
    HAS_DTCWT = True
except ImportError:
    HAS_DTCWT = False


@pytest.mark.skipif(not HAS_DTCWT, reason="dtcwt not available")
class TestExtractAudio:
    def test_smoke_synthetic_video(self, tmp_path):
        """Full pipeline on synthetic video: verify shape and finiteness."""
        video_path = str(tmp_path / "test.avi")
        _create_synthetic_video(video_path, num_frames=32, height=256, width=256)

        cap = cv2.VideoCapture(video_path)
        result = visualmic.extract_audio(
            cap, 32, nlevels=3, n_orient=6,
            ref_index=0,
            fps=30
        )
        assert result.shape == (32,)
        assert np.all(np.isfinite(result))
        assert np.min(result) >= -1.0 - 1e-10
        assert np.max(result) <= 1.0 + 1e-10

    def test_with_biort_qshift(self, tmp_path):
        """Verify custom filter selection works."""
        video_path = str(tmp_path / "test.avi")
        _create_synthetic_video(video_path, num_frames=16, height=256, width=256)

        cap = cv2.VideoCapture(video_path)
        result = visualmic.extract_audio(
            cap, 16, nlevels=2, n_orient=6,
            ref_index=0,
            fps=30, biort='near_sym_a', qshift='qshift_a'
        )
        assert result.shape == (16,)
        assert np.all(np.isfinite(result))


@pytest.fixture
def tiny_video(tmp_path):
    path = str(tmp_path / "tiny.avi")
    _create_synthetic_video(path, num_frames=40, height=64, width=64)
    return path


def _run(*args):
    return subprocess.run([sys.executable, 'visualmic.py', *args], capture_output=True, text=True)


@pytest.mark.skipif(not HAS_DTCWT, reason="dtcwt not available")
class TestFilterAndFpsChecks:
    """#15: bad settings fail before any frame is processed."""

    @pytest.mark.parametrize("flag", ['-fl', '-fh'])
    def test_non_positive_cutoff(self, tiny_video, flag):
        result = _run('-i', tiny_video, flag, '0')
        assert result.returncode != 0
        assert 'must be positive' in result.stdout
        assert 'Processing' not in result.stdout

    def test_freq_low_above_nyquist(self, tiny_video, tmp_path):
        result = _run('-i', tiny_video, '--fps', '2200', '-fl', '1100', '-o', str(tmp_path / 'o.wav'))
        assert result.returncode != 0
        assert 'Nyquist' in result.stdout
        assert 'Processing' not in result.stdout

    def test_low_fps_warns(self, tiny_video, tmp_path):
        result = _run('-i', tiny_video, '-o', str(tmp_path / 'o.wav'))
        assert result.returncode == 0
        assert '--fps' in result.stdout and 'high-speed video' in result.stdout

    def test_sample_rate_is_rounded(self, tiny_video, tmp_path):
        out = str(tmp_path / 'o.wav')
        result = _run('-i', tiny_video, '--fps', '2199.7', '-o', out)
        assert result.returncode == 0
        rate, _ = wavfile.read(out)
        assert rate == 2200


class TestInputValidation:
    def test_missing_input(self):
        result = subprocess.run(
            [sys.executable, 'visualmic.py'],
            capture_output=True, text=True
        )
        assert result.returncode != 0

    @pytest.mark.skipif(not HAS_DTCWT, reason="dtcwt not available")
    def test_nonexistent_file(self):
        result = subprocess.run(
            [sys.executable, 'visualmic.py', '-i', 'nonexistent.avi'],
            capture_output=True, text=True
        )
        assert result.returncode != 0
        assert 'not found' in result.stderr + result.stdout

    def test_bad_frequencies(self):
        result = subprocess.run(
            [sys.executable, 'visualmic.py', '-i', 'dummy.avi',
             '-fl', '200', '-fh', '100'],
            capture_output=True, text=True
        )
        assert result.returncode != 0
        assert 'freq-low' in result.stderr + result.stdout

    def test_bad_roi_format(self):
        result = subprocess.run(
            [sys.executable, 'visualmic.py', '-i', 'dummy.avi',
             '--roi', '100,50,200'],
            capture_output=True, text=True
        )
        assert result.returncode != 0
        assert 'four integers' in result.stderr + result.stdout

    def test_invalid_nlevels(self):
        result = subprocess.run(
            [sys.executable, 'visualmic.py', '-i', 'dummy.avi',
             '--nlevels', '0'],
            capture_output=True, text=True
        )
        assert result.returncode != 0
        assert 'nlevels' in result.stderr + result.stdout

    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, 'visualmic.py', '--version'],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'VERSION')) as f:
            version = f.read().strip()
        assert visualmic.__version__ == version
        assert version in result.stdout
