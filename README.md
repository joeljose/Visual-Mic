# Visual-Mic

When sound hits an object, it causes small vibrations of the object’s surface. Here we show how, using only high-speed video of the object, we can extract those minute vibrations and partially recover the sound that produced them, allowing us to turn everyday objects—a glass of water, a potted plant, a box of tissues, or a bag of chips—into visual microphones. </br>
The original project was done by MIT-CSAIL team. They have captured high-speed videos of packet of chips moving due to an audio clip of "Mary Had A Little Lamb "song. The video decomposition was done using a technique called Riesz Pyramids. In our project we use the same videos provided in the MIT-CSAIL website, but we use 2D Dual Tree Complex Wavelet Transform instead.</br>
The videos can be downloaded from [here](http://data.csail.mit.edu/vidmag/VisualMic/)

![](https://github.com/joeljose/assets/raw/master/Visual-Mic/vmic.png)

## Setting up visual mic

###  A. Setting up Python3(skip if already setup)

You can follow this link from [Youtube](https://www.youtube.com/watch?v=YYXdXT2l-Gg). This has a very concise explanation on how to setup python.

###  B. Rest of the setup

1. Clone the repo
   ```sh
   git clone https://github.com/joeljose/Visual-Mic.git
   ```
2. Navigate to "Visual-Mic" repo.
3. pip install all the python modules from requirements.txt(you should be in the "Visual-Mic" repository when you execute this command.)
   ```sh
   pip install requirements.txt
   ```
4. The video which is to be processed should be in the "Visual-Mic" repo, named as "testvid.avi". 

Now you can run visualmic.py 


## Details of Recovering Sound from Video

### Computing Local Motion Signals

We use phase variations in the 2D DTℂWT representation of the video V to
compute local motion. 2D DTℂWT breaks each frame of the video V (x, y,
t) into complex-valued sub-bands corresponding to different scales and
orientations. Each scale s and orientation *θ* is a complex image. We
can express them in terms of amplitude A and phase *ϕ* as:
*A*(*s*,*θ*,*x*,*y*,*t*)*e*<sup>*i*ϕ(*s*,*θ*,*x*,*y*,*t*)</sup>

Now to compute phase variation, we take local phases *ϕ* and subtract
them from the local phase of a reference frame *t*<sub>0</sub>(usually
the first frame).

*ϕ*<sub>*v*</sub>(*s*,*θ*,*x*,*y*,*t*) = *ϕ*(*s*,*θ*,*x*,*y*,*t*) − *ϕ*(*s*,*θ*,*x*,*y*,*t*<sub>0</sub>)

### Computing the Global Motion Signal

We then compute a spatially weighted average of the local motion
signals, for each scale s and orientation *θ* in the 2D DTℂWT
decomposition of the video, to produce a single motion signal *Φ*(s,
*θ*, t). Local motion signals in regions where there isn’t much texture
had ambiguous local phase information, and as a result motion signals in
these regions were noisy. So we perform a weighted average by taking the
square of the amplitude, since the amplitude gives a measure of texture
strength.

*Φ*(*s*,*θ*,*t*) = ∑<sub>*x*, *y*</sub>*A*(*s*,*θ*,*x*,*y*,*t*)<sup>2</sup>*ϕ*<sub>*v*</sub>(*s*,*θ*,*x*,*y*,*t*)

Our final global motion signal is obtained by averaging the
*Φ*(*s*,*θ*,*t*) over different scales and orientations.
*ŝ*(*t*) = ∑<sub>*s*, *θ*</sub>*Φ*(*s*,*θ*,*t*)
We finally scale this signal and center it to the range
\[−<!-- -->1,1\].

### Denoising
To denoise the ouput audio file we get from visualmic.py, we apply image based morphological filtering to the audio spectrograms, and then reconstruct audio from the processed spectrogram. Denoising had a lot of steps, so I've made it into a different project. Here is the link to the project https://github.com/joeljose/audio_denoising

## Follow Me
<a href="https://twitter.com/joelk1jose" target="_blank"><img class="ai-subscribed-social-icon" src="https://github.com/joeljose/assets/blob/master/images/tw.png" width="30"></a>
<a href="https://github.com/joeljose" target="_blank"><img class="ai-subscribed-social-icon" src="https://github.com/joeljose/assets/blob/master/images/gthb.png" width="30"></a>
<a href="https://www.linkedin.com/in/joel-jose-527b80102/" target="_blank"><img class="ai-subscribed-social-icon" src="https://github.com/joeljose/assets/blob/master/images/lnkdn.png" width="30"></a>

<h3 align="center">Show your support by starring the repository 🙂</h3>


