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

def soundfromvid(cap,frameCount,nlevels,orient,ref_no,ref_orient,ref_level):

	tr = dtcwt.Transform2d()
	ref_frame = None
	data = []

	for fc in range(frameCount):
		ret, raw_frame = cap.read()
		if not ret or raw_frame is None:
			print(f"Warning: could not read frame {fc}, stopping at {len(data)} frames")
			break
		gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

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

	# Parse the command-line arguments
	args = parser.parse_args()
	filename = args.input
	output_name = args.output

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
	orient=6
	ref_no=0
	ref_level=0
	ref_orient=0

	npTowav(soundfromvid(cap,frameCount,nlevels,orient,ref_no,ref_orient,ref_level), output_name, int(fps))


if __name__ == "__main__":
		main()


