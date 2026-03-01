import argparse
import os
import sys
import dtcwt
from scipy import signal
import numpy as np
import cv2
from scipy.io.wavfile import write

def maxTime(a,b):
		c = signal.correlate(a, b, mode='full')
		return np.argmax(c) - (len(b) - 1)

def npTowav(np_file, output_name, sps):
		# Samples per second(sampling frequency of the audio file)
		waveform_integers = np.int16(np_file * 32767)

		# Write the .wav file
		write(output_name, sps, waveform_integers)
		print(f"Output saved to {output_name}")

def soundfromvid(cap,frameCount,nlevels,orient,ref_no,ref_orient,ref_level,fps,freq_low=None,freq_high=None,roi=None):

	tr = dtcwt.Transform2d()
	ref_frame = None
	data = []

	for fc in range(frameCount):
		ret, raw_frame = cap.read()
		if not ret or raw_frame is None:
			print(f"Warning: could not read frame {fc}, stopping at {len(data)} frames")
			break
		gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
		if roi is not None:
			rx, ry, rw, rh = roi
			gray = gray[ry:ry+rh, rx:rx+rw]

		dtcwt_frame = tr.forward(gray, nlevels=nlevels)

		if fc == ref_no:
			ref_frame = dtcwt_frame

		if ref_frame is None:
			# Haven't reached reference frame yet, store zeros
			data.append(np.zeros((nlevels, orient)))
			continue

		row = np.zeros((nlevels, orient))
		for level in range(nlevels):
			for angle in range(orient):
				coeffs = dtcwt_frame.highpasses[level][:,:,angle]
				ref_coeffs = ref_frame.highpasses[level][:,:,angle]
				amp = np.abs(coeffs)
				phase = np.angle(coeffs)
				ref_phase = np.angle(ref_coeffs)
				phase_diff = np.angle(np.exp(1j * (phase - ref_phase)))
				row[level,angle] = np.sum(amp*amp * phase_diff)
		data.append(row)

	cap.release()

	if len(data) == 0:
		print("Error: no frames could be read from video")
		sys.exit(1)

	frameCount = len(data)
	data = np.array(data)
	print(f"Transform complete: {frameCount} frames processed")

	# Temporal bandpass filtering
	nyquist = fps / 2.0
	apply_filter = (freq_low is not None or freq_high is not None) and frameCount > 12

	if apply_filter:
		if freq_low is not None and freq_high is not None:
			if freq_low >= nyquist:
				print(f"Warning: freq_low ({freq_low} Hz) >= Nyquist ({nyquist} Hz), skipping filter")
				apply_filter = False
			else:
				freq_high_clamped = min(freq_high, nyquist * 0.99)
				sos = signal.butter(4, [freq_low / nyquist, freq_high_clamped / nyquist], btype='bandpass', output='sos')
				print(f"Applying bandpass filter: {freq_low}–{freq_high_clamped:.0f} Hz")
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
			for j in range(orient):
				data[:,i,j] = signal.sosfiltfilt(sos, data[:,i,j])

	shift_matrix=np.zeros((nlevels,orient))
	ref_vector=data[:,ref_level,ref_orient].reshape(-1)
	for i in range(nlevels):
		for j in range(orient):
			shift_matrix[i,j]=maxTime(ref_vector,data[:,i,j].reshape(-1))
	
	sound_raw=np.zeros(frameCount)
	for fc in range(frameCount):
		for i in range(nlevels):
			for j in range(orient):
				sound_raw[fc]+=data[fc-int(shift_matrix[i,j]),i,j]
	p_min = np.min(sound_raw)
	p_max = np.max(sound_raw)
	if p_max == p_min:
		print("Warning: no motion detected in video, output will be silent")
		sound_data = np.zeros_like(sound_raw)
	else:
		sound_data=((2*sound_raw)-(p_min+p_max))/(p_max-p_min)
	
	return (sound_data)


def main():
	# Create the argument parser
	parser = argparse.ArgumentParser(description='Motion Magnification using 2D DTCWT')

	# Add arguments
	parser.add_argument('-i', '--input', required=True, help='Specify input video path')
	parser.add_argument('-o', '--output', default='sound.wav', help='Specify output audio path (default: sound.wav)')
	parser.add_argument('-fl', '--freq-low', type=float, default=None, help='Lower cutoff frequency in Hz for temporal bandpass filter')
	parser.add_argument('-fh', '--freq-high', type=float, default=None, help='Upper cutoff frequency in Hz for temporal bandpass filter')
	parser.add_argument('--roi', type=str, default=None, help='Region of interest as x,y,w,h (e.g. --roi 100,50,200,150)')

	# Parse the command-line arguments
	args = parser.parse_args()
	filename = args.input
	output_name = args.output
	freq_low = args.freq_low
	freq_high = args.freq_high
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
	frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	if frameCount <= 0:
		print("Error: video has no frames")
		cap.release()
		sys.exit(1)

	if fps <= 0:
		print("Warning: could not determine FPS from video, defaulting to 30")
		fps = 30

	print(f"frameCount: {frameCount}, frameWidth: {frameWidth}, frameHeight: {frameHeight}, fps: {fps}")

	nlevels=3
	min_dim = 2 ** nlevels  # minimum dimension for DTCWT (8 pixels for 3 levels)

	if roi is not None:
		rx, ry, rw, rh = roi
		if rx < 0 or ry < 0 or rw <= 0 or rh <= 0:
			print("Error: ROI values must be non-negative and width/height must be positive")
			cap.release()
			sys.exit(1)
		if rx + rw > frameWidth or ry + rh > frameHeight:
			print(f"Error: ROI ({rx},{ry},{rw},{rh}) exceeds frame dimensions ({frameWidth}x{frameHeight})")
			cap.release()
			sys.exit(1)
		if rw < min_dim or rh < min_dim:
			print(f"Error: ROI dimensions ({rw}x{rh}) too small for {nlevels}-level DTCWT (minimum {min_dim}x{min_dim})")
			cap.release()
			sys.exit(1)
		print(f"Using ROI: x={rx}, y={ry}, w={rw}, h={rh}")

	orient=6
	ref_no=0
	ref_level=0
	ref_orient=0

	npTowav(soundfromvid(cap,frameCount,nlevels,orient,ref_no,ref_orient,ref_level,fps,freq_low,freq_high,roi), output_name, int(fps))


if __name__ == "__main__":
		main()


