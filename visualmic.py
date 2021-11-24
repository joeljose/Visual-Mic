import dtcwt
from scipy import ndimage
from scipy import signal
import numpy as np
import cv2
import cmath
from scipy.io.wavfile import write

def npTowav(np_file, output_name):
    # Samples per second(sampling frequency of the audio file)
    sps = 1
    waveform_integers = np.int16(file * 32767)

    # Write the .wav file
    write(output_name, sps, waveform_integers)

def soundfromvid(input_data,frameCount,nlevels,orient,ref_no,ref_orient,ref_level):

  tr = dtcwt.Transform2d()
  
  ref_frame= tr.forward(input_data[ref_no],nlevels =nlevels)
  
  data = np.zeros((frameCount,nlevels,orient))

  
  for fc in range(frameCount):
  
    frame = tr.forward(input_data[fc],nlevels =nlevels)


  
    for level in range(nlevels):
  
      xmax=len(frame.highpasses[level])
      ymax=len(frame.highpasses[level][0])
  
      for angle in range(orient):
  
        for x in range(xmax):
  
          for y in range(ymax):
  
            amp,phase=cmath.polar(frame.highpasses[level][x][y][angle])
            ref_phase=cmath.polar(ref_frame.highpasses[level][x][y][angle])[1]
  
            data[fc,level,angle]+=(amp*amp)*(phase-ref_phase)
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
  p_min = sound_raw[0]
  p_max = sound_raw[0]
  for i in range(1,frameCount):
    if sound_raw[i]<p_min:
      p_min=sound_raw[i]
    if sound_raw[i]>p_max:
      p_max=sound_raw[i]
  sound_data=((2*sound_raw)-(p_min+p_max))/(p_max-p_min)
  
  return (sound_data)


def main():
	cap = cv2.VideoCapture("testvid.avi")

	fps = cap.get(cv2.CAP_PROP_FPS)
	frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print(frameCount,frameWidth,frameHeight)

	nlevels=3
	orient=6
	ref_no=0
	ref_level=0
	ref_orient=0

	input_data = []


	fc = 0
	ret = True

	for fc in range(frameCount):

	  ret, frame = cap.read()
	  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	  input_data.append(gray)

	cap.release()
	print("Video Loaded")


	npTowav(soundfromvid(input_data,frameCount,nlevels,orient,ref_no,ref_orient,ref_level), "sound.wav")


if __name__ == "__main__":
    main()


