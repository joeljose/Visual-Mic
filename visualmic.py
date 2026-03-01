import argparse
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

def soundfromvid(input_data,frameCount,nlevels,orient,ref_no,ref_orient,ref_level):

	tr = dtcwt.Transform2d()
	ref_frame= tr.forward(input_data[ref_no],nlevels =nlevels)
	data = np.zeros((frameCount,nlevels,orient))

	for fc in range(frameCount):
		frame = tr.forward(input_data[fc],nlevels =nlevels)
		for level in range(nlevels):
			for angle in range(orient):
				coeffs = frame.highpasses[level][:,:,angle]
				ref_coeffs = ref_frame.highpasses[level][:,:,angle]
				amp = np.abs(coeffs)
				phase = np.angle(coeffs)
				ref_phase = np.angle(ref_coeffs)
				data[fc,level,angle] = np.sum(amp*amp * (phase - ref_phase))
	print("Transform complete")
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
	sound_data=((2*sound_raw)-(p_min+p_max))/(p_max-p_min)
	
	return (sound_data)


def main():
	# Create the argument parser
	parser = argparse.ArgumentParser(description='Motion Magnification using 2D DTCWT')

	# Add arguments
	parser.add_argument('-i', '--input', required=True, help='Specify input video path')

	# Parse the command-line arguments
	args = parser.parse_args()
	filename = args.input

	cap = cv2.VideoCapture(filename)

	fps = cap.get(cv2.CAP_PROP_FPS)
	frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	print(f"frameCount: {frameCount}, frameWidth: {frameWidth}, frameHeight: {frameHeight}")

	nlevels=3
	orient=6
	ref_no=0
	ref_level=0
	ref_orient=0

	input_data = []

	for fc in range(frameCount):
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		input_data.append(gray)

	cap.release()
	print("Video Loaded")


	npTowav(soundfromvid(input_data,frameCount,nlevels,orient,ref_no,ref_orient,ref_level), "sound.wav", int(fps))


if __name__ == "__main__":
		main()


