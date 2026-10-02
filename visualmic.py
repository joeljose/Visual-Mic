"""
Visual Microphone: Recover sound from video using 2D DTCWT.

Recovers sound from high-speed video by analyzing sub-pixel surface vibrations.
Uses the phase of complex wavelet coefficients to detect motion far too small
to see with the naked eye, then reconstructs an audible signal.

Based on: Davis et al., "The Visual Microphone: Passive Recovery of Sound
from Video", ACM Transactions on Graphics (SIGGRAPH 2014).
"""

__version__ = "3.1.0"

import argparse
import itertools
import os
import sys
import time

from scipy import signal
import numpy as np
import cv2
from scipy.io.wavfile import write


def format_duration(seconds):
	seconds = int(seconds)
	if seconds < 60:
		return f"{seconds}s"
	elif seconds < 3600:
		return f"{seconds // 60}m {seconds % 60}s"
	else:
		h = seconds // 3600
		m = (seconds % 3600) // 60
		s = seconds % 60
		return f"{h}h {m}m {s}s"


# Directions of the six DTCWT orientations (15, 45, ..., 165 deg). The 105-165
# deg bands respond to motion like -75..-15 deg: their phase rises with
# rightward motion, so their vertical response is inverted.
ORIENT_ANGLES = np.deg2rad([15, 45, 75, -75, -45, -15])


def print_progress(done, expected, start_time):
	"""Print frames done, percent and ETA. expected is the container's frame
	count, which can be wrong, so it never drops below done."""
	expected = max(expected, done)
	elapsed = time.time() - start_time
	rate = done / elapsed if elapsed > 0 else 0
	remaining = (expected - done) / rate if rate > 0 else 0
	print(f"Processing: {done}/{expected} frames ({100 * done // expected}%) | Elapsed: {format_duration(elapsed)} | ETA: {format_duration(remaining)}")


def warn_if_count_differs(reported, decoded):
	if decoded != reported:
		print(f"Warning: the video reports {reported} frames, but {decoded} could be decoded. Using all {decoded}.")


def default_freq_low(fps):
	"""High-pass cutoff used unless -fl or --no-filter is given.

	Davis et al. 2014 (Sec. 3.3) high-pass at 20-100 Hz, typically 1/20 of
	Nyquist, to remove low-frequency drift and noise that isn't audio.
	"""
	return min(max(fps / 40.0, 20.0), 100.0)


def denoise_spectral(samples, fs, alpha=2.0, beta=0.05):
	"""Spectral subtraction (Boll 1979) with a stationary noise estimate.

	The noise power of each frequency is its median over the whole clip:
	sound comes and goes, while hum, light flicker and sensor noise stay.
	alpha over-subtracts to suppress residual noise; beta is the gain floor.
	"""
	n = len(samples)
	nperseg = min(256, n)
	_, _, Z = signal.stft(samples, fs=fs, nperseg=nperseg)
	power = np.abs(Z) ** 2
	noise = np.median(power, axis=1, keepdims=True)
	gain = np.sqrt(np.maximum(1 - alpha * noise / (power + 1e-20), beta ** 2))
	_, out = signal.istft(Z * gain, fs=fs, nperseg=nperseg)
	return np.pad(out, (0, max(0, n - len(out))))[:n]


def save_wav(samples, output_name, sample_rate):
	waveform_integers = np.int16(samples * 32767)
	write(output_name, sample_rate, waveform_integers)
	print(f"Output saved to {output_name}")


def vibration_direction(dx, dy, fs):
	"""Unit vector of the dominant vibration direction in the image plane.

	Principal axis of the 2D motion after removing stationary noise, so that
	hum and light flicker don't decide the direction. Sign is fixed (x >= 0).
	"""
	motion = np.stack([denoise_spectral(dx, fs), denoise_spectral(dy, fs)])
	_, vectors = np.linalg.eigh(np.cov(motion))
	u = vectors[:, -1]
	return u if u[0] >= 0 else -u


def postprocess_phase_signals(phase_signals, fps, freq_low=None, freq_high=None, denoise=False):
	"""Turn per-band phase signals of shape (frames, levels, orientations) into audio."""
	frame_count = len(phase_signals)

	# Temporal bandpass filtering
	nyquist = fps / 2.0
	apply_filter = (freq_low is not None or freq_high is not None) and frame_count > 1

	if apply_filter:
		if freq_low is not None and freq_high is not None:
			if freq_low >= nyquist:
				print(f"Warning: freq_low ({freq_low} Hz) >= Nyquist ({nyquist} Hz), skipping filter")
				apply_filter = False
			else:
				freq_high_clamped = min(freq_high, nyquist * 0.99)
				sos = signal.butter(4, [freq_low / nyquist, freq_high_clamped / nyquist], btype='bandpass', output='sos')
				print(f"Applying bandpass filter: {freq_low} to {freq_high_clamped:.0f} Hz")
		elif freq_low is not None:
			if freq_low >= nyquist:
				print(f"Warning: freq_low ({freq_low} Hz) >= Nyquist ({nyquist} Hz), skipping filter")
				apply_filter = False
			else:
				sos = signal.butter(4, freq_low / nyquist, btype='highpass', output='sos')
				print(f"Applying highpass filter: {freq_low} Hz")
		else:
			freq_high_clamped = min(freq_high, nyquist * 0.99)
			sos = signal.butter(4, freq_high_clamped / nyquist, btype='lowpass', output='sos')
			print(f"Applying lowpass filter: {freq_high_clamped:.0f} Hz")

	if apply_filter:
		# sosfiltfilt pads both ends by 3 * ntaps samples (scipy's default).
		# Shorter clips get a shorter pad instead of a ValueError.
		ntaps = 2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
		padlen = min(3 * ntaps, frame_count - 1)
		phase_signals = signal.sosfiltfilt(sos, phase_signals, axis=0, padlen=padlen)

	# Each orientation band measures the motion projected onto its direction.
	# Combine them into x/y motion and take the dominant vibration direction.
	# Davis et al. instead align bands by cross-correlation; on real footage
	# (MIT Chips2) that picked arbitrary lags and cost 7 dB SNR, and plain
	# summing cancels vertical motion, which half the orientations see inverted.
	dx = (phase_signals * np.cos(ORIENT_ANGLES)).sum(axis=(1, 2))
	dy = (phase_signals * np.sin(ORIENT_ANGLES)).sum(axis=(1, 2))
	u = vibration_direction(dx, dy, fps)
	sound_raw = u[0] * dx + u[1] * dy

	if denoise:
		sound_raw = denoise_spectral(sound_raw, fps)

	p_min = np.min(sound_raw)
	p_max = np.max(sound_raw)
	if p_max == p_min:
		print("Warning: no motion detected in video, output will be silent")
		sound_data = np.zeros_like(sound_raw)
	else:
		sound_data = ((2 * sound_raw) - (p_min + p_max)) / (p_max - p_min)

	return sound_data


def extract_audio(cap, frame_count, nlevels, n_orient, fps, freq_low=None, freq_high=None, roi=None, biort='near_sym_b', qshift='qshift_b', denoise=False):
	import dtcwt
	transform = dtcwt.Transform2d(biort=biort, qshift=qshift)
	prev_conj = None
	phase_signals = []
	progress_interval = max(1, frame_count // 10)
	start_time = time.time()

	# Read until the video ends: the container's frame count can be wrong
	for fc in itertools.count():
		ret, raw_frame = cap.read()
		if not ret or raw_frame is None:
			break
		gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
		if roi is not None:
			rx, ry, rw, rh = roi
			gray = gray[ry:ry+rh, rx:rx+rw]

		highpasses = transform.forward(gray, nlevels=nlevels).highpasses
		if prev_conj is None:
			prev_conj = [np.conj(h) for h in highpasses]

		# Phase change since the previous frame, weighted by amplitude squared
		frame_phases = np.zeros((nlevels, n_orient))
		for level in range(nlevels):
			coeffs = highpasses[level]
			amp = np.abs(coeffs)
			phase_diff = np.angle(coeffs * prev_conj[level])
			frame_phases[level, :] = np.sum(amp * amp * phase_diff, axis=(0, 1))
		phase_signals.append(frame_phases)
		prev_conj = [np.conj(h) for h in highpasses]

		if (fc + 1) % progress_interval == 0:
			print_progress(fc + 1, frame_count, start_time)

	cap.release()
	warn_if_count_differs(frame_count, len(phase_signals))

	if len(phase_signals) == 0:
		print("Error: no frames could be read from video")
		sys.exit(1)

	frame_count = len(phase_signals)
	phase_signals = accumulate_phase(phase_signals)
	elapsed = time.time() - start_time
	print(f"Transform complete: {frame_count} frames in {format_duration(elapsed)}")

	return postprocess_phase_signals(phase_signals, fps, freq_low, freq_high, denoise)


def accumulate_phase(phase_changes):
	"""Add up frame-to-frame phase changes into phase relative to the first frame.

	Phase is only known modulo 2*pi. Measured against one fixed frame it wraps
	once the surface drifts by about half a wavelength of a band. Between two
	consecutive frames at audio frame rates the change is tiny, so it never
	wraps, and the running sum follows any amount of drift (#17).
	"""
	return np.cumsum(np.array(phase_changes, dtype=np.float64), axis=0)


def estimate_vram(batch_size, height, width, nlevels):
	"""Estimate peak GPU VRAM usage in bytes.

	Peak occurs during batched forward DTCWT: input frames plus
	transform intermediates (~15x overhead per frame).
	"""
	frame_bytes = height * width * 4  # float32
	dtcwt_overhead = 15  # empirical: forward transform intermediates
	batch_vram = batch_size * frame_bytes * dtcwt_overhead
	pytorch_overhead = 300 * 1024 * 1024  # ~300 MB for PyTorch + filter weights
	return batch_vram + pytorch_overhead


def extract_audio_gpu(cap, frame_count, nlevels, n_orient, fps, freq_low=None, freq_high=None, roi=None, batch_size=16, biort='near_sym_b', qshift='qshift_b', denoise=False):
	import torch
	from pytorch_wavelets import DTCWTForward

	device = torch.device('cuda')
	xfm = DTCWTForward(J=nlevels, biort=biort, qshift=qshift).to(device)
	print(f"GPU mode: {torch.cuda.get_device_name(0)}, batch_size={batch_size}")

	prev = None
	phase_signals = []
	progress_interval = max(1, frame_count // 10)
	last_report = 0
	start_time = time.time()

	def process_batch(frames):
		nonlocal prev
		batch_np = np.stack(frames)[:, np.newaxis, :, :]
		try:
			batch_tensor = torch.from_numpy(batch_np).to(device)
			Yl, Yh = xfm(batch_tensor)
		except RuntimeError as e:
			if 'out of memory' in str(e).lower():
				print(f"Error: GPU out of memory with batch_size={batch_size}. Try a smaller --batch-size.")
				cap.release()
				sys.exit(1)
			raise

		if prev is None:
			prev = [Yh[level][:1] for level in range(nlevels)]

		batch_phases = np.zeros((len(frames), nlevels, n_orient))
		for level in range(nlevels):
			# Yh[level] shape: (N, 1, 6, H, W, 2), last dim is real/imag.
			# Each frame is compared with the one before it.
			hp = Yh[level]
			before = torch.cat([prev[level], hp[:-1]])

			c_real = hp[..., 0]
			c_imag = hp[..., 1]
			r_real = before[..., 0]
			r_imag = before[..., 1]

			# Conjugate multiply: (c + id)(a - ib) = (ca+db) + i(da-cb)
			prod_real = c_real * r_real + c_imag * r_imag
			prod_imag = c_imag * r_real - c_real * r_imag

			phase_diff = torch.atan2(prod_imag, prod_real)
			amp_sq = c_real * c_real + c_imag * c_imag

			# Sum over spatial dims (H, W) -> (N, 1, 6)
			weighted = (amp_sq * phase_diff).sum(dim=(-2, -1))
			batch_phases[:, level, :] = weighted[:, 0, :].cpu().numpy()

		phase_signals.extend(batch_phases)
		prev = [Yh[level][-1:].clone() for level in range(nlevels)]

		del batch_tensor, Yl, Yh

	batch_frames = []

	# Read until the video ends: the container's frame count can be wrong
	while True:
		ret, raw_frame = cap.read()
		if not ret or raw_frame is None:
			break
		gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
		if roi is not None:
			rx, ry, rw, rh = roi
			gray = gray[ry:ry+rh, rx:rx+rw]
		batch_frames.append(gray.astype(np.float32))

		if len(batch_frames) == batch_size:
			process_batch(batch_frames)
			batch_frames = []
			if len(phase_signals) >= last_report + progress_interval:
				print_progress(len(phase_signals), frame_count, start_time)
				last_report = len(phase_signals)

	if batch_frames:
		process_batch(batch_frames)

	cap.release()
	warn_if_count_differs(frame_count, len(phase_signals))

	if len(phase_signals) == 0:
		print("Error: no frames could be read from video")
		sys.exit(1)

	frame_count = len(phase_signals)
	phase_signals = accumulate_phase(phase_signals)
	elapsed = time.time() - start_time
	print(f"Transform complete: {frame_count} frames in {format_duration(elapsed)}")

	return postprocess_phase_signals(phase_signals, fps, freq_low, freq_high, denoise)


def main():
	parser = argparse.ArgumentParser(description='Visual Microphone: Recover sound from video using 2D DTCWT')

	parser.add_argument(
		'--version', action='version',
		version=f'%(prog)s {__version__}'
	)
	parser.add_argument('-i', '--input', required=True, help='Input video path')
	parser.add_argument('-o', '--output', default='sound.wav', help='Output audio path (default: sound.wav)')
	parser.add_argument('-fl', '--freq-low', type=float, default=None, help='Lower cutoff frequency in Hz for temporal bandpass filter (default: fps/40, clamped to 20-100 Hz)')
	parser.add_argument('--no-filter', action='store_true', help='Disable the default high-pass filter (raw phase signals)')
	parser.add_argument('--denoise', action='store_true', help='Spectral subtraction of stationary noise (hum, light flicker, sensor noise)')
	parser.add_argument('-fh', '--freq-high', type=float, default=None, help='Upper cutoff frequency in Hz for temporal bandpass filter')
	parser.add_argument('--fps', type=float, default=None, help='Override video frame rate (Hz) for audio output sample rate')
	parser.add_argument('--roi', type=str, default=None, help='Region of interest as x,y,w,h (e.g. --roi 100,50,200,150)')
	parser.add_argument('--gpu', action='store_true', help='Use GPU-accelerated DTCWT (requires CUDA and pytorch_wavelets)')
	parser.add_argument('--batch-size', type=int, default=16, help='Frames per GPU batch (default: 16, GPU mode only)')
	parser.add_argument('--nlevels', type=int, default=3, help='Number of DTCWT decomposition levels (default: 3)')
	parser.add_argument('--biort', default='near_sym_b', help='DTCWT biorthogonal filter (default: near_sym_b)')
	parser.add_argument('--qshift', default='qshift_b', help='DTCWT quarter-shift filter (default: qshift_b)')

	args = parser.parse_args()
	pipeline_start = time.time()

	filename = args.input
	output_name = args.output
	freq_low = args.freq_low
	freq_high = args.freq_high
	for flag, value in (('--freq-low', freq_low), ('--freq-high', freq_high)):
		if value is not None and value <= 0:
			print(f"Error: {flag} must be positive (got {value:g})")
			sys.exit(1)
	if freq_low is not None and freq_high is not None and freq_low >= freq_high:
		print(f"Error: freq-low ({freq_low} Hz) must be less than freq-high ({freq_high} Hz)")
		sys.exit(1)
	nlevels = args.nlevels
	if nlevels < 1:
		print("Error: --nlevels must be >= 1")
		sys.exit(1)
	roi = None
	if args.roi is not None:
		try:
			parts = [int(p) for p in args.roi.split(',')]
			if len(parts) != 4:
				raise ValueError
			roi = tuple(parts)
		except ValueError:
			print("Error: --roi must be four integers: x,y,w,h (e.g. --roi 100,50,200,150)")
			sys.exit(1)

	if args.gpu:
		try:
			import torch
			if not torch.cuda.is_available():
				print("Error: --gpu requires CUDA but no GPU is available")
				sys.exit(1)
		except ImportError:
			print("Error: --gpu requires PyTorch (pip install torch)")
			sys.exit(1)
		try:
			import pytorch_wavelets  # noqa: F401
		except ImportError:
			print("Error: --gpu requires pytorch_wavelets (pip install git+https://github.com/fbcotter/pytorch_wavelets.git)")
			sys.exit(1)
	else:
		try:
			import dtcwt  # noqa: F401
		except ImportError:
			print("Error: CPU mode requires dtcwt (pip install dtcwt)")
			sys.exit(1)

	if not os.path.isfile(filename):
		print(f"Error: file '{filename}' not found")
		sys.exit(1)

	cap = cv2.VideoCapture(filename)
	if not cap.isOpened():
		print(f"Error: could not open '{filename}' as video")
		sys.exit(1)

	fps = cap.get(cv2.CAP_PROP_FPS)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	if frame_count <= 0:
		print("Error: video has no frames")
		cap.release()
		sys.exit(1)

	if fps <= 0:
		print("Warning: could not determine FPS from video, defaulting to 30")
		fps = 30

	if args.fps is not None:
		if args.fps <= 0:
			print("Error: --fps must be positive")
			cap.release()
			sys.exit(1)
		print(f"Overriding video FPS ({fps}) with --fps {args.fps}")
		fps = args.fps

	print(f"frame_count: {frame_count}, frame_width: {frame_width}, frame_height: {frame_height}, fps: {fps}")

	# Check the filter against the frame rate now, not after hours of processing
	nyquist = fps / 2.0
	if fps < 500:
		print(f"Warning: at {fps:g} fps the audio can only hold frequencies up to {nyquist:g} Hz. "
			"The Visual Microphone needs high-speed video. If the camera ran faster than the file "
			"says (the MIT samples report about 30 fps but were shot at 2200), pass the real rate with --fps.")
	max_cutoff = nyquist * 0.99
	if freq_low is not None and freq_low >= max_cutoff:
		print(f"Error: --freq-low ({freq_low:g} Hz) must be below {max_cutoff:.0f} Hz (99% of the Nyquist frequency at {fps:g} fps)")
		cap.release()
		sys.exit(1)
	if freq_high is not None and freq_high > max_cutoff:
		print(f"Note: --freq-high ({freq_high:g} Hz) is above 99% of the Nyquist frequency; using {max_cutoff:.0f} Hz")

	min_dim = 2 ** nlevels

	if roi is not None:
		rx, ry, rw, rh = roi
		if rx < 0 or ry < 0 or rw <= 0 or rh <= 0:
			print("Error: ROI values must be non-negative and width/height must be positive")
			cap.release()
			sys.exit(1)
		if rx + rw > frame_width or ry + rh > frame_height:
			print(f"Error: ROI ({rx},{ry},{rw},{rh}) exceeds frame dimensions ({frame_width}x{frame_height})")
			cap.release()
			sys.exit(1)
		if rw < min_dim or rh < min_dim:
			print(f"Error: ROI dimensions ({rw}x{rh}) too small for {nlevels}-level DTCWT (minimum {min_dim}x{min_dim})")
			cap.release()
			sys.exit(1)
		print(f"Using ROI: x={rx}, y={ry}, w={rw}, h={rh}")

	n_orient = 6

	if freq_low is None and not args.no_filter:
		default_low = default_freq_low(fps)
		if default_low < max_cutoff and (freq_high is None or default_low < freq_high):
			freq_low = default_low

	if args.gpu:
		import torch
		proc_h = roi[3] if roi else frame_height
		proc_w = roi[2] if roi else frame_width
		required = estimate_vram(args.batch_size, proc_h, proc_w, nlevels)
		free, total = torch.cuda.mem_get_info(0)
		required_gb = required / (1024 ** 3)
		free_gb = free / (1024 ** 3)
		total_gb = total / (1024 ** 3)
		print(f"  Estimated VRAM needed: {required_gb:.1f} GB")
		print(f"  GPU VRAM available:    {free_gb:.1f} GB / {total_gb:.1f} GB")
		if required > free * 0.7:
			print(
				f"\nWarning: estimated VRAM ({required_gb:.1f} GB) exceeds 70% of "
				f"available ({free_gb:.1f} GB).\n"
				f"  Suggestions:\n"
				f"  - Reduce --batch-size (current: {args.batch_size})\n"
				f"  - Use --roi to crop to a smaller region\n"
				f"  - Remove --gpu to use CPU mode",
				file=sys.stderr
			)
		sound_data = extract_audio_gpu(cap, frame_count, nlevels, n_orient, fps, freq_low, freq_high, roi, args.batch_size, args.biort, args.qshift, args.denoise)
	else:
		sound_data = extract_audio(cap, frame_count, nlevels, n_orient, fps, freq_low, freq_high, roi, args.biort, args.qshift, args.denoise)

	save_wav(sound_data, output_name, round(fps))
	print(f"Total time: {format_duration(time.time() - pipeline_start)}")


if __name__ == "__main__":
	main()
