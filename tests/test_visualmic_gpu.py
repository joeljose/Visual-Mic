"""Unit tests for visualmic.py — GPU path (CUDA only)."""

import os
import sys

import cv2
import numpy as np
import pytest

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import visualmic


pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


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

        xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b').cuda()
        batch = torch.randn(4, 1, 256, 256, device='cuda')
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

        xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b').cuda()
        batch = torch.randn(2, 1, 256, 256, device='cuda')
        Yl, Yh = xfm(batch)

        assert torch.isfinite(Yl).all()
        for level in range(3):
            assert torch.isfinite(Yh[level]).all()

    def test_custom_filters(self):
        """Verify custom biort/qshift filters work."""
        from pytorch_wavelets import DTCWTForward

        xfm = DTCWTForward(J=2, biort='near_sym_a', qshift='qshift_a').cuda()
        batch = torch.randn(2, 1, 256, 256, device='cuda')
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
            ref_index=0, ref_orient=0, ref_level=0,
            fps=30, batch_size=16
        )
        assert result.shape == (32,)
        assert np.all(np.isfinite(result))
        assert np.min(result) >= -1.0 - 1e-10
        assert np.max(result) <= 1.0 + 1e-10

    def test_with_custom_filters(self, tmp_path):
        """Verify GPU path works with custom filter selection."""
        video_path = str(tmp_path / "test.avi")
        _create_synthetic_video(video_path, num_frames=16, height=256, width=256)

        cap = cv2.VideoCapture(video_path)
        result = visualmic.extract_audio_gpu(
            cap, 16, nlevels=2, n_orient=6,
            ref_index=0, ref_orient=0, ref_level=0,
            fps=30, batch_size=8,
            biort='near_sym_a', qshift='qshift_a'
        )
        assert result.shape == (16,)
        assert np.all(np.isfinite(result))


class TestEstimateVram:
    def test_arithmetic(self):
        result = visualmic.estimate_vram(16, 256, 256, 3)
        expected = 16 * 256 * 256 * 4 * 15 + 300 * 1024 * 1024
        assert result == expected

    def test_scales_with_batch_size(self):
        small = visualmic.estimate_vram(8, 256, 256, 3)
        large = visualmic.estimate_vram(32, 256, 256, 3)
        assert large > small
