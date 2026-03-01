import argparse
import os
import sys
import time
from scipy import signal
from scipy import ndimage
import numpy as np
import cv2
from scipy.io.wavfile import read as read_wav
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


def find_best_shift(a, b):
	correlation = signal.correlate(a, b, mode='full')
	return np.argmax(correlation) - (len(b) - 1)


def save_wav(samples, output_name, sample_rate):
	waveform_integers = np.int16(samples * 32767)
	write(output_name, sample_rate, waveform_integers)
	print(f"Output saved to {output_name}")


def denoise_spectral(samples, fs, noise_duration=0.1):
	f, t, Zxx = signal.stft(samples, fs=fs, nperseg=512)
	magnitude = np.abs(Zxx)
	phase = np.angle(Zxx)

	# Estimate noise from first noise_duration seconds
	noise_frames = max(1, int(noise_duration * fs / (512 // 4)))
	noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

	# Subtract noise, floor at zero
	clean_mag = np.maximum(magnitude - noise_profile, 0.0)

	# Reconstruct with original phase
	clean_Zxx = clean_mag * np.exp(1j * phase)
	_, reconstructed = signal.istft(clean_Zxx, fs=fs, nperseg=512)

	# Normalize to [-1, 1]
	peak = np.max(np.abs(reconstructed))
	if peak > 0:
		reconstructed = reconstructed / peak
	return reconstructed


def denoise_morphological(samples, fs, threshold=20, amp=10):
	f, t, Zxx = signal.stft(samples, fs=fs, nperseg=512)
	magnitude = np.abs(Zxx)

	# Convert to grayscale (0-255)
	mag_max = np.max(magnitude)
	if mag_max == 0:
		return samples
	gray = magnitude * (255.0 / mag_max)

	# Binary threshold
	mask = gray >= threshold

	# Morphological erosion then dilation
	mask = ndimage.binary_erosion(mask, iterations=1)
	mask = ndimage.binary_dilation(mask, iterations=2)

	# Apply mask: amplify signal, attenuate noise
	masked_Zxx = np.where(mask, Zxx * amp, Zxx / amp)

	_, reconstructed = signal.istft(masked_Zxx, fs=fs, nperseg=512)

	# Normalize to [-1, 1]
	peak = np.max(np.abs(reconstructed))
	if peak > 0:
		reconstructed = reconstructed / peak
	return reconstructed


def postprocess_phase_signals(phase_signals, frame_count, nlevels, n_orient, ref_level, ref_orient, fps, freq_low=None, freq_high=None):
	# Temporal bandpass filtering
	nyquist = fps / 2.0
	apply_filter = (freq_low is not None or freq_high is not None) and frame_count > 12

	if apply_filter:
		if freq_low is not None and freq_high is not None:
			if freq_low >= nyquist:
				print(f"Warning: freq_low ({freq_low} Hz) >= Nyquist ({nyquist} Hz), skipping filter")
				apply_filter = False
			else:
				freq_high_clamped = min(freq_high, nyquist * 0.99)
				sos = signal.butter(4, [freq_low / nyquist, freq_high_clamped / nyquist], btype='bandpass', output='sos')
				print(f"Applying bandpass filter: {freq_low}\u2013{freq_high_clamped:.0f} Hz")
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
		for i in range(nlevels):
			for j in range(n_orient):
				phase_signals[:, i, j] = signal.sosfiltfilt(sos, phase_signals[:, i, j])

	shift_matrix = np.zeros((nlevels, n_orient))
	ref_vector = phase_signals[:, ref_level, ref_orient].reshape(-1)
	for i in range(nlevels):
		for j in range(n_orient):
			shift_matrix[i, j] = find_best_shift(ref_vector, phase_signals[:, i, j].reshape(-1))

	sound_raw = np.zeros(frame_count)
	for i in range(nlevels):
		for j in range(n_orient):
			sound_raw += np.roll(phase_signals[:, i, j], int(shift_matrix[i, j]))

	p_min = np.min(sound_raw)
	p_max = np.max(sound_raw)
	if p_max == p_min:
		print("Warning: no motion detected in video, output will be silent")
		sound_data = np.zeros_like(sound_raw)
	else:
		sound_data = ((2 * sound_raw) - (p_min + p_max)) / (p_max - p_min)

	return sound_data


def extract_audio(cap, frame_count, nlevels, n_orient, ref_index, ref_orient, ref_level, fps, freq_low=None, freq_high=None, roi=None):
	import dtcwt
	transform = dtcwt.Transform2d()
	ref_conj = None
	phase_signals = []
	progress_interval = max(1, frame_count // 10)
	start_time = time.time()

	for fc in range(frame_count):
		ret, raw_frame = cap.read()
		if not ret or raw_frame is None:
			print(f"Warning: could not read frame {fc}, stopping at {len(phase_signals)} frames")
			break
		gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
		if roi is not None:
			rx, ry, rw, rh = roi
			gray = gray[ry:ry+rh, rx:rx+rw]

		dtcwt_frame = transform.forward(gray, nlevels=nlevels)

		if fc == ref_index:
			ref_conj = [np.conj(dtcwt_frame.highpasses[level]) for level in range(nlevels)]

		if ref_conj is None:
			phase_signals.append(np.zeros((nlevels, n_orient)))
			continue

		frame_phases = np.zeros((nlevels, n_orient))
		for level in range(nlevels):
			coeffs = dtcwt_frame.highpasses[level]
			amp = np.abs(coeffs)
			phase_diff = np.angle(coeffs * ref_conj[level])
			frame_phases[level, :] = np.sum(amp * amp * phase_diff, axis=(0, 1))
		phase_signals.append(frame_phases)

		if (fc + 1) % progress_interval == 0 or fc == frame_count - 1:
			elapsed = time.time() - start_time
			rate = (fc + 1) / elapsed if elapsed > 0 else 0
			remaining = (frame_count - fc - 1) / rate if rate > 0 else 0
			print(f"Processing: {fc + 1}/{frame_count} frames ({100 * (fc + 1) // frame_count}%) | Elapsed: {format_duration(elapsed)} | ETA: {format_duration(remaining)}")

	cap.release()

	if len(phase_signals) == 0:
		print("Error: no frames could be read from video")
		sys.exit(1)

	frame_count = len(phase_signals)
	phase_signals = np.array(phase_signals)
	elapsed = time.time() - start_time
	print(f"Transform complete: {frame_count} frames in {format_duration(elapsed)}")

	return postprocess_phase_signals(phase_signals, frame_count, nlevels, n_orient, ref_level, ref_orient, fps, freq_low, freq_high)


def extract_audio_gpu(cap, frame_count, nlevels, n_orient, ref_index, ref_orient, ref_level, fps, freq_low=None, freq_high=None, roi=None, batch_size=16):
	import torch
	from pytorch_wavelets import DTCWTForward

	device = torch.device('cuda')
	xfm = DTCWTForward(J=nlevels, biort='near_sym_b', qshift='qshift_b').to(device)
	print(f"GPU mode: {torch.cuda.get_device_name(0)}, batch_size={batch_size}")

	ref_coeffs = None
	phase_signals = []
	progress_interval = max(1, frame_count // 10)
	frames_read = 0
	last_report = 0
	start_time = time.time()

	batch_frames = []

	for fc in range(frame_count):
		ret, raw_frame = cap.read()
		if not ret or raw_frame is None:
			print(f"Warning: could not read frame {fc}, stopping at {frames_read} frames")
			break
		gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
		if roi is not None:
			rx, ry, rw, rh = roi
			gray = gray[ry:ry+rh, rx:rx+rw]

		batch_frames.append(gray.astype(np.float32))
		frames_read += 1

		if len(batch_frames) == batch_size or fc == frame_count - 1:
			n_in_batch = len(batch_frames)
			batch_start_fc = fc - n_in_batch + 1

			batch_np = np.stack(batch_frames)[:, np.newaxis, :, :]
			try:
				batch_tensor = torch.from_numpy(batch_np).to(device)
				Yl, Yh = xfm(batch_tensor)
			except RuntimeError as e:
				if 'out of memory' in str(e).lower():
					print(f"Error: GPU out of memory with batch_size={batch_size}. Try a smaller --batch-size.")
					cap.release()
					sys.exit(1)
				raise

			# Extract reference coefficients if reference frame is in this batch
			if ref_coeffs is None and ref_index >= batch_start_fc and ref_index <= fc:
				ref_pos = ref_index - batch_start_fc
				ref_coeffs = [Yh[level][ref_pos:ref_pos+1].clone() for level in range(nlevels)]

			if ref_coeffs is None:
				# Haven't seen reference frame yet
				for _ in range(n_in_batch):
					phase_signals.append(np.zeros((nlevels, n_orient)))
			else:
				batch_phases = np.zeros((n_in_batch, nlevels, n_orient))
				for level in range(nlevels):
					# Yh[level] shape: (N, 1, 6, H, W, 2), last dim is real/imag
					hp = Yh[level]
					ref_hp = ref_coeffs[level]

					c_real = hp[..., 0]
					c_imag = hp[..., 1]
					r_real = ref_hp[..., 0]
					r_imag = ref_hp[..., 1]

					# Conjugate multiply: (c + id)(a - ib) = (ca+db) + i(da-cb)
					prod_real = c_real * r_real + c_imag * r_imag
					prod_imag = c_imag * r_real - c_real * r_imag

					phase_diff = torch.atan2(prod_imag, prod_real)
					amp_sq = c_real * c_real + c_imag * c_imag

					# Sum over spatial dims (H, W) -> (N, 1, 6)
					weighted = (amp_sq * phase_diff).sum(dim=(-2, -1))
					batch_phases[:, level, :] = weighted[:, 0, :].cpu().numpy()

				for i in range(n_in_batch):
					if batch_start_fc + i < ref_index:
						phase_signals.append(np.zeros((nlevels, n_orient)))
					else:
						phase_signals.append(batch_phases[i])

			del batch_tensor, Yl, Yh
			batch_frames = []

			if frames_read >= last_report + progress_interval or fc == frame_count - 1:
				elapsed = time.time() - start_time
				rate = frames_read / elapsed if elapsed > 0 else 0
				remaining = (frame_count - frames_read) / rate if rate > 0 else 0
				print(f"Processing: {frames_read}/{frame_count} frames ({100 * frames_read // frame_count}%) | Elapsed: {format_duration(elapsed)} | ETA: {format_duration(remaining)}")
				last_report = frames_read

	cap.release()

	if len(phase_signals) == 0:
		print("Error: no frames could be read from video")
		sys.exit(1)

	frame_count = len(phase_signals)
	phase_signals = np.array(phase_signals)
	elapsed = time.time() - start_time
	print(f"Transform complete: {frame_count} frames in {format_duration(elapsed)}")

	return postprocess_phase_signals(phase_signals, frame_count, nlevels, n_orient, ref_level, ref_orient, fps, freq_low, freq_high)


def main():
	parser = argparse.ArgumentParser(description='Visual Microphone: Recover sound from video using 2D DTCWT')

	parser.add_argument('-i', '--input', default=None, help='Specify input video path')
	parser.add_argument('-o', '--output', default='sound.wav', help='Specify output audio path (default: sound.wav)')
	parser.add_argument('-fl', '--freq-low', type=float, default=None, help='Lower cutoff frequency in Hz for temporal bandpass filter')
	parser.add_argument('-fh', '--freq-high', type=float, default=None, help='Upper cutoff frequency in Hz for temporal bandpass filter')
	parser.add_argument('--fps', type=float, default=None, help='Override video frame rate (Hz) for audio output sample rate')
	parser.add_argument('--roi', type=str, default=None, help='Region of interest as x,y,w,h (e.g. --roi 100,50,200,150)')
	parser.add_argument('--gpu', action='store_true', help='Use GPU-accelerated DTCWT (requires CUDA and pytorch_wavelets)')
	parser.add_argument('--batch-size', type=int, default=16, help='Frames per GPU batch (default: 16, GPU mode only)')
	parser.add_argument('--denoise', choices=['spectral', 'morphological'], default=None, help='Audio denoising method (applied after reconstruction)')
	parser.add_argument('--denoise-input', type=str, default=None, help='Denoise an existing WAV file instead of processing video')

	args = parser.parse_args()
	pipeline_start = time.time()

	# Standalone denoise mode
	if args.denoise_input is not None:
		if args.denoise is None:
			print("Error: --denoise-input requires --denoise {spectral,morphological}")
			sys.exit(1)
		if not os.path.isfile(args.denoise_input):
			print(f"Error: file '{args.denoise_input}' not found")
			sys.exit(1)
		sr, wav_data = read_wav(args.denoise_input)
		samples = wav_data.astype(np.float64) / 32767.0
		print(f"Loaded {args.denoise_input}: {len(samples)} samples, {sr} Hz, {len(samples)/sr:.2f}s")
		print(f"Applying {args.denoise} denoising...")
		if args.denoise == 'spectral':
			samples = denoise_spectral(samples, sr)
		else:
			samples = denoise_morphological(samples, sr)
		print("Denoising complete")
		save_wav(samples, args.output, sr)
		print(f"Total time: {format_duration(time.time() - pipeline_start)}")
		return

	# Full pipeline mode — require -i
	if args.input is None:
		print("Error: -i/--input is required (or use --denoise-input for standalone denoising)")
		sys.exit(1)

	filename = args.input
	output_name = args.output
	freq_low = args.freq_low
	freq_high = args.freq_high
	if freq_low is not None and freq_high is not None and freq_low >= freq_high:
		print(f"Error: freq-low ({freq_low} Hz) must be less than freq-high ({freq_high} Hz)")
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
			import pytorch_wavelets
		except ImportError:
			print("Error: --gpu requires pytorch_wavelets (pip install git+https://github.com/fbcotter/pytorch_wavelets.git)")
			sys.exit(1)
	else:
		try:
			import dtcwt
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

	nlevels = 3
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
	ref_index = 0
	ref_level = 0
	ref_orient = 0

	if args.gpu:
		sound_data = extract_audio_gpu(cap, frame_count, nlevels, n_orient, ref_index, ref_orient, ref_level, fps, freq_low, freq_high, roi, args.batch_size)
	else:
		sound_data = extract_audio(cap, frame_count, nlevels, n_orient, ref_index, ref_orient, ref_level, fps, freq_low, freq_high, roi)

	if args.denoise:
		print(f"Applying {args.denoise} denoising...")
		if args.denoise == 'spectral':
			sound_data = denoise_spectral(sound_data, int(fps))
		else:
			sound_data = denoise_morphological(sound_data, int(fps))
		print("Denoising complete")

	save_wav(sound_data, output_name, int(fps))
	print(f"Total time: {format_duration(time.time() - pipeline_start)}")


if __name__ == "__main__":
	main()
