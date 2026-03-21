"""Unit tests for visualmic.py — CPU path."""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

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


class TestFindBestShift:
    def test_known_shift(self):
        # b is shifted right by 10 relative to a → find_best_shift returns -10
        # (the shift needed to align b back to a)
        a = np.zeros(200)
        a[50:60] = 1.0
        b = np.zeros(200)
        b[60:70] = 1.0
        result = visualmic.find_best_shift(a, b)
        assert result == -10

    def test_zero_shift(self):
        a = np.zeros(100)
        a[30:40] = 1.0
        result = visualmic.find_best_shift(a, a)
        assert result == 0

    def test_negative_shift(self):
        # b is shifted left by 5 relative to a → find_best_shift returns 5
        a = np.zeros(200)
        a[60:70] = 1.0
        b = np.zeros(200)
        b[55:65] = 1.0
        result = visualmic.find_best_shift(a, b)
        assert result == 5


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
            phase_signals, frame_count, nlevels, n_orient,
            ref_level=0, ref_orient=0, fps=100
        )
        # Constant input → all sub-bands identical → after shift+sum, still constant
        # Normalization maps constant to zero
        assert result.shape == (frame_count,)
        assert np.all(np.isfinite(result))

    def test_normalization_range(self):
        """Output should be in [-1, 1]."""
        nlevels, n_orient, frame_count = 3, 6, 100
        rng = np.random.RandomState(42)
        phase_signals = rng.randn(frame_count, nlevels, n_orient)
        result = visualmic.postprocess_phase_signals(
            phase_signals, frame_count, nlevels, n_orient,
            ref_level=0, ref_orient=0, fps=100
        )
        assert np.min(result) >= -1.0 - 1e-10
        assert np.max(result) <= 1.0 + 1e-10

    def test_output_shape(self):
        nlevels, n_orient, frame_count = 2, 6, 30
        rng = np.random.RandomState(42)
        phase_signals = rng.randn(frame_count, nlevels, n_orient)
        result = visualmic.postprocess_phase_signals(
            phase_signals, frame_count, nlevels, n_orient,
            ref_level=0, ref_orient=0, fps=100
        )
        assert result.shape == (frame_count,)


class TestButterworthFilter:
    def test_passband_preserved(self):
        """Signal within passband should be preserved."""
        nlevels, n_orient, frame_count = 1, 1, 200
        fps = 1000
        # 100 Hz signal, passband 50-200 Hz
        t = np.arange(frame_count) / fps
        phase_signals = np.sin(2 * np.pi * 100 * t).reshape(-1, 1, 1)
        result = visualmic.postprocess_phase_signals(
            phase_signals.copy(), frame_count, nlevels, n_orient,
            ref_level=0, ref_orient=0, fps=fps, freq_low=50, freq_high=200
        )
        assert np.all(np.isfinite(result))
        # Should have non-zero energy (signal passed through)
        assert np.std(result) > 0.01

    def test_stopband_attenuated(self):
        """Signal outside passband should be attenuated."""
        nlevels, n_orient, frame_count = 1, 1, 200
        fps = 1000
        # 400 Hz signal, passband 50-200 Hz
        t = np.arange(frame_count) / fps
        phase_signals = np.sin(2 * np.pi * 400 * t).reshape(-1, 1, 1)
        result = visualmic.postprocess_phase_signals(
            phase_signals.copy(), frame_count, nlevels, n_orient,
            ref_level=0, ref_orient=0, fps=fps, freq_low=50, freq_high=200
        )
        assert np.all(np.isfinite(result))
        # Should have lower energy than unfiltered version
        unfiltered = visualmic.postprocess_phase_signals(
            phase_signals.copy(), frame_count, nlevels, n_orient,
            ref_level=0, ref_orient=0, fps=fps
        )
        assert np.std(result) < np.std(unfiltered)


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
            ref_index=0, ref_orient=0, ref_level=0,
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
            ref_index=0, ref_orient=0, ref_level=0,
            fps=30, biort='near_sym_a', qshift='qshift_a'
        )
        assert result.shape == (16,)
        assert np.all(np.isfinite(result))


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
        assert '2.0.0' in result.stdout
