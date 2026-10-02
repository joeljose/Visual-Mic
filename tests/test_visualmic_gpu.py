"""Unit tests for visualmic.py, PyTorch path. Runs on CUDA when available, else on the CPU."""

import os
import sys

import cv2
import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import visualmic

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_wavelets")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _create_synthetic_video(path, num_frames=32, height=256, width=256):
    """Create a synthetic video with random noise for testing."""
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    rng = np.random.RandomState(42)
    for _ in range(num_frames):
        frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


class TestGpuForwardPass:
    def test_dtcwt_forward_shapes(self):
        """Verify DTCWTForward produces expected shapes."""
        from pytorch_wavelets import DTCWTForward

        xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b').to(DEVICE)
        batch = torch.randn(4, 1, 256, 256, device=DEVICE)
        Yl, Yh = xfm(batch)

        assert Yl.shape[0] == 4
        assert len(Yh) == 3
        for level in range(3):
            assert Yh[level].shape[0] == 4
            assert Yh[level].shape[2] == 6  # 6 orientations
            assert Yh[level].shape[-1] == 2  # real/imag

    def test_dtcwt_forward_finite(self):
        """Verify all outputs are finite."""
        from pytorch_wavelets import DTCWTForward

        xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b').to(DEVICE)
        batch = torch.randn(2, 1, 256, 256, device=DEVICE)
        Yl, Yh = xfm(batch)

        assert torch.isfinite(Yl).all()
        for level in range(3):
            assert torch.isfinite(Yh[level]).all()

    def test_custom_filters(self):
        """Verify custom biort/qshift filters work."""
        from pytorch_wavelets import DTCWTForward

        xfm = DTCWTForward(J=2, biort='near_sym_a', qshift='qshift_a').to(DEVICE)
        batch = torch.randn(2, 1, 256, 256, device=DEVICE)
        Yl, Yh = xfm(batch)
        assert torch.isfinite(Yl).all()


class TestExtractAudioGpu:
    def test_smoke_synthetic_video(self, tmp_path):
        """Full GPU pipeline on synthetic video: verify shape and finiteness."""
        video_path = str(tmp_path / "test.avi")
        _create_synthetic_video(video_path, num_frames=32, height=256, width=256)

        cap = cv2.VideoCapture(video_path)
        result = visualmic.extract_audio_gpu(
            cap, 32, nlevels=3, n_orient=6,
            fps=30, batch_size=16, device=DEVICE
        )
        assert result.shape == (32,)
        assert np.all(np.isfinite(result))
        assert np.min(result) >= -1.0 - 1e-10
        assert np.max(result) <= 1.0 + 1e-10

    @pytest.mark.parametrize("reported", [20, 50], ids=["under-reported", "over-reported"])
    def test_output_length_follows_decoded_frames(self, tmp_path, reported):
        """#16: the last partial batch is processed and no frame is dropped."""
        video_path = str(tmp_path / "test.avi")
        _create_synthetic_video(video_path, num_frames=40, height=64, width=64)

        cap = cv2.VideoCapture(video_path)
        result = visualmic.extract_audio_gpu(
            cap, reported, nlevels=3, n_orient=6,
            fps=2200, batch_size=16, device=DEVICE
        )
        assert result.shape == (40,)

    def test_with_custom_filters(self, tmp_path):
        """Verify GPU path works with custom filter selection."""
        video_path = str(tmp_path / "test.avi")
        _create_synthetic_video(video_path, num_frames=16, height=256, width=256)

        cap = cv2.VideoCapture(video_path)
        result = visualmic.extract_audio_gpu(
            cap, 16, nlevels=2, n_orient=6,
            fps=30, batch_size=8, device=DEVICE,
            biort='near_sym_a', qshift='qshift_a'
        )
        assert result.shape == (16,)
        assert np.all(np.isfinite(result))


class TestEstimateVram:
    def test_arithmetic(self):
        result = visualmic.estimate_vram(16, 256, 256)
        expected = 16 * 256 * 256 * 4 * 15 + 300 * 1024 * 1024
        assert result == expected

    def test_scales_with_batch_size(self):
        small = visualmic.estimate_vram(8, 256, 256)
        large = visualmic.estimate_vram(32, 256, 256)
        assert large > small


def test_pytorch_path_matches_numpy_path(tmp_path):
    """The PyTorch and NumPy paths give the same audio, also under drift."""
    pytest.importorskip("dtcwt")
    import io
    import contextlib
    from scipy import ndimage

    rng = np.random.RandomState(0)
    tex = ndimage.gaussian_filter(rng.rand(128, 128), 2.0) * 2000
    t = np.arange(300) / 2200
    shift = 0.3 * np.sin(2 * np.pi * 300 * t) + 4 * t / t[-1]  # tone plus 4 px of drift
    kx = np.fft.fftfreq(128)[None, :]
    path = str(tmp_path / "drift.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (128, 128))
    for d in shift:
        frame = np.real(np.fft.ifft2(np.fft.fft2(tex) * np.exp(-2j * np.pi * kx * d)))
        gray = np.clip(frame - frame.min() + 20, 0, 255).astype(np.uint8)
        out.write(np.repeat(gray[:, :, None], 3, axis=2))
    out.release()

    with contextlib.redirect_stdout(io.StringIO()):
        a = visualmic.extract_audio(cv2.VideoCapture(path), 300, 3, 6, 2200, 55.0)
        b = visualmic.extract_audio_gpu(cv2.VideoCapture(path), 300, 3, 6, 2200, 55.0, batch_size=16, device=DEVICE)
    assert abs(np.corrcoef(a, b)[0, 1]) > 0.999


def test_cli_device_flag(tmp_path):
    """--gpu --device runs the PyTorch path end to end, including the VRAM check on CUDA."""
    import subprocess
    video_path = str(tmp_path / "test.avi")
    _create_synthetic_video(video_path, num_frames=20, height=64, width=64)
    out = str(tmp_path / "o.wav")
    result = subprocess.run(
        [sys.executable, 'visualmic.py', '--gpu', '--device', DEVICE, '-i', video_path, '-o', out, '--fps', '2200'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists(out)
