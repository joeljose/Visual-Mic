import argparse
import os
import sys
import dtcwt
from scipy import signal
import numpy as np
import cv2
from scipy.io.wavfile import write


def find_best_shift(a, b):
	correlation = signal.correlate(a, b, mode='full')
	return np.argmax(correlation) - (len(b) - 1)


def save_wav(samples, output_name, sample_rate):
	waveform_integers = np.int16(samples * 32767)
	write(output_name, sample_rate, waveform_integers)
	print(f"Output saved to {output_name}")


def extract_audio(cap, frame_count, nlevels, n_orient, ref_index, ref_orient, ref_level, fps, freq_low=None, freq_high=None, roi=None):
	transform = dtcwt.Transform2d()
	ref_conj = None
	phase_signals = []
	progress_interval = max(1, frame_count // 10)

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
			print(f"Processing: {fc + 1}/{frame_count} frames ({100 * (fc + 1) // frame_count}%)")

	cap.release()

	if len(phase_signals) == 0:
		print("Error: no frames could be read from video")
		sys.exit(1)

	frame_count = len(phase_signals)
	phase_signals = np.array(phase_signals)
	print(f"Transform complete: {frame_count} frames processed")

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


def main():
	parser = argparse.ArgumentParser(description='Visual Microphone: Recover sound from video using 2D DTCWT')

	parser.add_argument('-i', '--input', required=True, help='Specify input video path')
	parser.add_argument('-o', '--output', default='sound.wav', help='Specify output audio path (default: sound.wav)')
	parser.add_argument('-fl', '--freq-low', type=float, default=None, help='Lower cutoff frequency in Hz for temporal bandpass filter')
	parser.add_argument('-fh', '--freq-high', type=float, default=None, help='Upper cutoff frequency in Hz for temporal bandpass filter')
	parser.add_argument('--roi', type=str, default=None, help='Region of interest as x,y,w,h (e.g. --roi 100,50,200,150)')

	args = parser.parse_args()
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

	save_wav(
		extract_audio(cap, frame_count, nlevels, n_orient, ref_index, ref_orient, ref_level, fps, freq_low, freq_high, roi),
		output_name,
		int(fps)
	)


if __name__ == "__main__":
	main()
