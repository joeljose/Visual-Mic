# Visual-Mic

[![CI](https://github.com/joeljose/Visual-Mic/actions/workflows/ci.yml/badge.svg)](https://github.com/joeljose/Visual-Mic/actions/workflows/ci.yml)

Play music next to a bag of chips and the bag shakes. Not much: its surface moves by about a thousandth of a millimetre, which in a video is a hundredth of a pixel or less. You can't see that. But a computer can measure it in every frame of a high-speed video, and a list of tiny movements, one per frame, is a sound recording. Play it back and you hear the music.

This repository does that. It is a Python implementation of the **Visual Microphone** of [Davis et al. (MIT CSAIL, SIGGRAPH 2014)](https://people.csail.mit.edu/mrub/VisualMic/). It reads a high-speed video of an object, measures how the object moves from frame to frame, and writes the motion out as a WAV file.

The original paper measured motion with a complex steerable pyramid. This project uses the **2D Dual-Tree Complex Wavelet Transform (DTCWT)**, which does the same job with far fewer coefficients. On MIT's own test videos it recovers the music about as well as the paper did. On the video of a potted plant, its output matches MIT's published result on two of three quality scores (see [Results](#results)).

![](assets/vmic.png)

---

## Contents

- [Quick start](#quick-start)
- [Getting good results](#getting-good-results)
- [Command-line reference](#command-line-reference)
- [How it works](#how-it-works)
  - [1. How big is the motion?](#1-how-big-is-the-motion)
  - [2. Phase measures position](#2-phase-measures-position)
  - [3. Local phase: complex wavelets](#3-local-phase-complex-wavelets)
  - [4. The dual-tree complex wavelet transform](#4-the-dual-tree-complex-wavelet-transform)
  - [5. Measuring motion from one frame to the next](#5-measuring-motion-from-one-frame-to-the-next)
  - [6. Many pixels, one number per band](#6-many-pixels-one-number-per-band)
  - [7. Six orientations, one direction](#7-six-orientations-one-direction)
  - [8. Cleaning up the signal](#8-cleaning-up-the-signal)
  - [9. Writing the sound file](#9-writing-the-sound-file)
  - [The whole pipeline](#the-whole-pipeline)
  - [Notation](#notation)
- [The original method, and what we changed](#the-original-method-and-what-we-changed)
- [Inside the code](#inside-the-code)
- [Results](#results)
- [Development](#development)
- [Related work](#related-work)
- [References](#references)

---

## Quick start

### 1. Get a test video

MIT published the videos from the paper at [data.csail.mit.edu/vidmag/VisualMic](https://data.csail.mit.edu/vidmag/VisualMic/). Each one comes with the sound that was played to the object (`*-input.wav`) and MIT's own recovered sound (`*-recovered.wav`), so you can compare. The videos are large:

| Video | Object | Frames | Size |
|---|---|---|---|
| `Chips2-2200Hz-Mary_MIDI-input.avi` | bag of chips | 38,083 at 704x400 | 11.6 GB |
| `Plant-2200Hz-Mary_MIDI-input.avi` | potted plant | 38,986 at 704x400 | 13.0 GB |
| `Chips1-2200Hz-Mary_Had-input.avi` | bag of chips | 22,859 at 704x704 | 14.2 GB |

All three were filmed at 2200 frames per second while "Mary Had a Little Lamb" played nearby. Chips2 is the smallest, so start there. Download it and its `input.wav` from the `Results/` folder of that page.

> The AVI files say they run at about 30 fps. They don't. They were filmed at 2200 fps and saved with the wrong rate in the file, so always pass `--fps 2200` with these videos. The program warns you when the frame rate looks too low for sound.

### 2. Run it

With a GPU. You need the NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Chips2 takes about 4 minutes on a laptop RTX 4050:

```bash
git clone https://github.com/joeljose/Visual-Mic.git
cd Visual-Mic
./docker-build-gpu.sh
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v /path/to/videos:/data \
    visual-mic-gpu:latest \
    --gpu -i /data/Chips2-2200Hz-Mary_MIDI-input.avi -o /data/sound.wav \
    --fps 2200 --denoise
```

Without a GPU, Chips2 takes about 6 minutes on a 6-core laptop:

```bash
./docker-build.sh
docker run --rm --user "$(id -u):$(id -g)" -v /path/to/videos:/data \
    visual-mic:latest \
    -i /data/Chips2-2200Hz-Mary_MIDI-input.avi -o /data/sound.wav --fps 2200 --denoise
```

The `--user` option makes the output file belong to you rather than to the container.

Without Docker, on Python 3.11 (the version the images and CI use):

```bash
pip install -r requirements.txt
python visualmic.py -i Chips2-2200Hz-Mary_MIDI-input.avi -o sound.wav --fps 2200 --denoise
```

### 3. Listen

`sound.wav` has one audio sample per video frame, so its sample rate is 2200 Hz. Some audio players refuse rates that low. Convert it to a normal rate first:

```bash
ffmpeg -i sound.wav -ar 16000 sound_16k.wav
```

### 4. Score it (optional)

Compare your result with the sound that was really played:

```bash
python scripts/eval_audio.py sound.wav --ref Chips2-2200Hz-Mary_MIDI-input.wav
```

[Results](#results) explains the three numbers it prints.

---

## Getting good results

- **Pass the true frame rate with `--fps`.** The sample rate of the output and the default filter both depend on it.
- **Add `--denoise` when you want to listen.** It removes steady hum and light flicker. Leave it off when you want the raw motion signal.
- **Crop to the object with `--roi x,y,w,h`** when much of the frame is something else. Time grows with the area processed, and background pixels only add noise.
- **Expect frequencies up to half the frame rate.** At 2200 fps that is 1100 Hz: enough for a tune or the pitch of a voice, not for clear consonants. [Section 9](#9-writing-the-sound-file) explains why.
- **Light, thin objects work best.** A bag of chips or a leaf moves a lot for a given sound. A brick hardly moves at all.

---

## Command-line reference

```bash
python visualmic.py -i INPUT [-o OUTPUT] [options]
```

| Option | Default | What it does |
|---|---|---|
| `-i`, `--input` | required | Input video |
| `-o`, `--output` | `sound.wav` | Output WAV file. It is checked before processing starts, so a bad path fails at once |
| `--fps` | from the video | The true frame rate in Hz. Sets the sample rate of the output |
| `-fl`, `--freq-low` | $`f_s/40`$, kept within 20 to 100 Hz | Low cutoff of the filter in Hz ([section 8](#8-cleaning-up-the-signal)) |
| `-fh`, `--freq-high` | none | High cutoff of the filter in Hz |
| `--no-filter` | off | Turn off the default high-pass filter |
| `--denoise` | off | Remove steady noise by spectral subtraction ([section 8](#8-cleaning-up-the-signal)) |
| `--roi` | whole frame | Process only the rectangle `x,y,w,h` |
| `--nlevels` | 3 | Number of DTCWT levels ([section 4](#4-the-dual-tree-complex-wavelet-transform)) |
| `--biort` | `near_sym_b` | DTCWT filter for level 1: `antonini`, `legall`, `near_sym_a` or `near_sym_b` |
| `--qshift` | `qshift_b` | DTCWT filter for levels 2 and up: `qshift_06`, `qshift_a`, `qshift_b`, `qshift_c` or `qshift_d` |
| `--jobs` | physical cores, at most 4 | Worker processes for the CPU path |
| `--gpu` | off | Use the PyTorch path (needs PyTorch and `pytorch_wavelets`) |
| `--device` | `cuda` | PyTorch device for `--gpu`: `cuda`, `cuda:N`, or `cpu` to run the PyTorch path without a GPU |
| `--batch-size` | 16 | Frames per batch on the PyTorch path. Lower it if the GPU runs out of memory |
| `--version` | | Print the version and exit |

Errors and warnings go to stderr, so you can pipe the normal output elsewhere.

---

## How it works

This section builds the method from the ground up. It assumes you know what a sine wave and a complex number are, and nothing about wavelets.

### 1. How big is the motion?

Sound is a pressure wave in the air. When it reaches an object, it pushes on the surface, and the surface moves back and forth with the sound. Davis et al. measured this with a laser vibrometer. In their calibrated tests, at sound levels up to $`95\,\text{dB}`$, the surfaces moved by up to about a micrometre, a thousandth of a millimetre.

How big is that in a video? It depends on the lens and the distance. In the paper's videos the motion was between $`0.002`$ and $`0.029`$ pixels (RMS). If a pixel were a metre wide, the object would move by a few centimetres.

Can we measure motion that small? Not by tracking a pixel. The brightness of one pixel barely changes when an edge moves by a hundredth of a pixel, and the sensor noise is bigger than that change. We need a way to measure position that uses many pixels at once and is very sensitive to small shifts. Phase is that measure.

### 2. Phase measures position

Take a one-dimensional picture: a row of pixels with brightness $`I(x)`$. Shift it to the right by a distance $`d`$. The new picture is

```math
I_d(x) = I(x - d).
```

Now write the picture as a sum of sine waves, its Fourier transform $`\hat I(\omega)`$, where $`\omega`$ is the spatial frequency in radians per pixel. The shift theorem says what happens to each wave:

```math
\hat I_d(\omega) = \hat I(\omega)\, e^{-i \omega d}.
```

Every wave keeps its strength, $`\lvert \hat I_d(\omega) \rvert = \lvert \hat I(\omega) \rvert`$. Only its phase changes, by

```math
\Delta\phi = -\omega d .
```

That is the whole idea. A shift of the picture becomes a change of phase, and the change is proportional to the shift. Measure the phase change, divide by $`-\omega`$, and you have the displacement.

How sensitive is it? Take a wave with a period of $`8`$ pixels, so $`\omega = 2\pi/8 \approx 0.785\,\text{rad/px}`$. A shift of $`0.01`$ pixels changes its phase by $`0.785 \times 0.01 \approx 0.0079\,\text{rad}`$. Small, but not lost in the noise of a single pixel, because every pixel the wave covers contributes to it.

### 3. Local phase: complex wavelets

The Fourier transform has one problem: each of its waves covers the whole picture. If the left half of the frame moves and the right half doesn't, a Fourier phase mixes the two. We want a phase that belongs to one small patch of the picture.

A **complex wavelet** gives exactly that. It is a short wave packet, a few periods long, with a real part shaped like a cosine and an imaginary part shaped like a sine. Slide it over the picture. At each position $`\mathbf{p}`$, multiply it with the patch under it and add up, which gives one complex number:

```math
c(\mathbf{p}) = A(\mathbf{p})\, e^{i \phi(\mathbf{p})} .
```

The **amplitude** $`A`$ says how much the patch looks like the wave: large on a strong edge or texture, near zero on a blank wall. The **phase** $`\phi`$ says where, within one period of the wave, that texture sits. Move the texture and $`\phi`$ changes as in [section 2](#2-phase-measures-position), but now locally.

In two dimensions the wave has a direction as well as a size. Its spatial frequency is a vector $`\boldsymbol{\omega}`$, and a shift by the vector $`\mathbf{d}`$ changes the phase by

```math
\Delta\phi \approx -\, \boldsymbol{\omega} \cdot \mathbf{d} .
```

The dot product means each wavelet only sees the part of the motion along its own direction. A wavelet with vertical stripes, so that $`\boldsymbol{\omega}`$ points sideways, feels sideways motion and is blind to vertical motion. That is why we need wavelets at several orientations.

### 4. The dual-tree complex wavelet transform

We need complex wavelets at several sizes and orientations, at every position of every frame. The **dual-tree complex wavelet transform** (DTCWT), developed by Nick Kingsbury, computes them efficiently. Selesnick, Baraniuk and Kingsbury give a readable introduction [[4]](#ref-dtcwt).

An ordinary discrete wavelet transform uses real wavelets. Their coefficients wobble up and down as a picture shifts, even by part of a pixel, so they make poor phase meters. The DTCWT runs two ordinary wavelet transforms side by side, two "trees", with filters designed so that the second tree's wavelets are very nearly the Hilbert transform of the first tree's. In plain terms: where one tree has a cosine-like wavelet, the other has the matching sine-like one. Take the first tree as the real part and the second as the imaginary part, and you get complex wavelets that behave like the wave packets of [section 3](#3-local-phase-complex-wavelets).

In two dimensions each **level** $`\ell`$ of the transform gives six complex sub-bands, one for each **orientation**. Their stripes are tilted at about $`\pm 15^\circ`$, $`\pm 45^\circ`$ and $`\pm 75^\circ`$. Each level halves the resolution, so the wavelets double in size:

| Level $`\ell`$ | Sub-band size | What it sees |
|---|---|---|
| 1 (finest) | $`H/2 \times W/2`$ | fine texture, sharp edges |
| 2 | $`H/4 \times W/4`$ | medium-scale pattern |
| 3 | $`H/8 \times W/8`$ | broad structure |

We use three levels by default, so every frame gives $`3 \times 6 = 18`$ sub-bands. All of them see the same motion, each through a different size and direction of wavelet.

The 2D DTCWT has only four times as many coefficients as the picture has pixels. The complex steerable pyramid of the original paper needs about 21 times as many with eight orientations [[3]](#ref-riesz). Fewer coefficients means less work per frame.

### 5. Measuring motion from one frame to the next

Write $`c_{\ell,k,t}(\mathbf{p})`$ for the coefficient at level $`\ell`$, orientation $`k`$ and position $`\mathbf{p}`$ in frame $`t`$. How do we get its phase change?

Multiplying one complex number by the conjugate of another subtracts their phases. So

```math
\Delta\phi_{\ell,k,t}(\mathbf{p}) = \arg\!\left( c_{\ell,k,t}(\mathbf{p}) \; \overline{c_{\ell,k,t-1}(\mathbf{p})} \right)
```

is the phase change from frame $`t-1`$ to frame $`t`$. Here $`\overline{c}`$ is the complex conjugate, and $`\arg`$ returns an angle in $`(-\pi, \pi]`$.

That last detail matters. Phase is only known up to whole turns: $`\phi`$ and $`\phi + 2\pi`$ look the same. If the true change were $`3.5\,\text{rad}`$, $`\arg`$ would report $`3.5 - 2\pi \approx -2.78\,\text{rad}`$, a jump the wrong way.

The original paper measures every frame against frame $`0`$. Over a long video the object, or the camera, drifts slowly. Once it has drifted by about half a wavelet period from where it was in frame $`0`$, the phase difference passes $`\pi`$ and jumps by a whole turn, and the sound gets a sudden step in it.

Between two **neighbouring** frames, though, the change is tiny: at 2200 frames per second nothing moves half a wavelength in $`0.45\,\text{ms}`$. So we measure each frame against the one before it, where no jump can happen, and add the changes up:

```math
\phi_{\ell,k,t}(\mathbf{p}) - \phi_{\ell,k,0}(\mathbf{p}) = \sum_{s=1}^{t} \Delta\phi_{\ell,k,s}(\mathbf{p}) .
```

This follows any amount of drift. In a synthetic test with $`10`$ pixels of drift, measuring against frame $`0`$ gave an SNR of $`-20\,\text{dB}`$, and measuring frame to frame gave $`15.9\,\text{dB}`$.

### 6. Many pixels, one number per band

Each sub-band has thousands of positions, and each gives its own noisy phase change. How do we combine them into one number per frame?

Not every position deserves the same trust. Picture a coefficient as an arrow of length $`A`$, with a little random noise added to its tip. A long arrow hardly turns when its tip is nudged. A short one can swing right round. To make that precise, let the noise $`n`$ added to $`c`$ be complex with average power $`\sigma^2`$. The phase error is then about

```math
\delta\phi \approx \frac{\operatorname{Im}\!\left( n\, e^{-i\phi} \right)}{A},
\qquad
\operatorname{Var}(\delta\phi) \approx \frac{\sigma^2}{2A^2} .
```

The standard way to combine independent measurements with different noise is to weight each one by the inverse of its variance. Here that weight is proportional to $`A^2`$. So each band gets one number per frame,

```math
\Delta\Phi_{\ell,k}(t) = \sum_{\mathbf{p}} \bigl\lvert c_{\ell,k,t}(\mathbf{p}) \bigr\rvert^{2} \, \Delta\phi_{\ell,k,t}(\mathbf{p}) ,
```

and a running sum over the frames,

```math
\Phi_{\ell,k}(t) = \sum_{s=1}^{t} \Delta\Phi_{\ell,k}(s), \qquad \Phi_{\ell,k}(0) = 0 .
```

Textured patches count a lot and blank patches hardly at all. This is the same $`A^2`$ weighting that Davis et al. use. We don't divide by the sum of the weights, because the output is rescaled at the end anyway.

### 7. Six orientations, one direction

We now have 18 signals $`\Phi_{\ell,k}(t)`$, one per band. They follow the same motion, but not in the same way. From [section 3](#3-local-phase-complex-wavelets), each band measures the motion projected onto its own direction:

```math
\Phi_{\ell,k}(t) \approx -\, g_{\ell,k}\; \mathbf{u}_k \cdot \mathbf{d}(t),
\qquad
\mathbf{u}_k = (\cos\theta_k,\ \sin\theta_k) ,
```

where $`\mathbf{d}(t)`$ is the displacement of the object and $`g_{\ell,k}`$ is a positive gain that depends on the band. For the DTCWT, the directions that behave this way are

```math
\theta_k \in \{\, 15^\circ,\ 45^\circ,\ 75^\circ,\ -75^\circ,\ -45^\circ,\ -15^\circ \,\} .
```

The DTCWT calls its last three bands $`105^\circ`$, $`135^\circ`$ and $`165^\circ`$, but they respond to motion as if they pointed at $`-75^\circ`$, $`-45^\circ`$ and $`-15^\circ`$. You can check this directly: for vertical motion the six bands come out with signs $`(+,+,+,-,-,-)`$.

That is why simply adding up all 18 signals fails. For vertical motion, three orientations see the motion one way and three see it the other way, and they cancel. In our synthetic test, plain addition turned a vertical vibration of $`0.01`$ pixels into an SNR of $`-19.5\,\text{dB}`$, which is just noise.

So we put the projections back together, the way you would read a vector off several tilted rulers. Weight each band by the components of its direction:

```math
d_x(t) = \sum_{\ell}\sum_{k} \cos\theta_k \; \Phi_{\ell,k}(t),
\qquad
d_y(t) = \sum_{\ell}\sum_{k} \sin\theta_k \; \Phi_{\ell,k}(t) .
```

Why does this work? Put the model above into these sums. The angles come in pairs $`\pm\theta`$, so the cross terms $`\cos\theta_k \sin\theta_k`$ cancel. With equal gains, $`d_x`$ is then proportional to the true horizontal motion and $`d_y`$ to the true vertical motion. For this set of angles the two factors are even equal, $`\sum_k \cos^2\theta_k = \sum_k \sin^2\theta_k = 3`$, so neither direction is favoured.

Finally, an object vibrating to sound moves mostly back and forth along one line. We find that line as the main axis of the points $`(d_x(t), d_y(t))`$. Form the $`2 \times 2`$ covariance matrix

```math
C = \frac{1}{N-1} \sum_{t} \bigl(\mathbf{m}(t) - \bar{\mathbf{m}}\bigr)\bigl(\mathbf{m}(t) - \bar{\mathbf{m}}\bigr)^{\top},
\qquad
\mathbf{m}(t) = \begin{pmatrix} d_x(t) \\ d_y(t) \end{pmatrix},
```

take its eigenvector $`\mathbf{v}`$ with the largest eigenvalue, and project onto it:

```math
s(t) = \mathbf{v}^{\top}\mathbf{m}(t) = v_x\, d_x(t) + v_y\, d_y(t) .
```

One refinement: $`C`$ is computed after steady noise has been removed from $`d_x`$ and $`d_y`$, with the method of [section 8](#8-cleaning-up-the-signal). Otherwise a strong hum, which says nothing about the vibration, could pull the axis its way. The sign of $`\mathbf{v}`$ is chosen so that $`v_x \ge 0`$. The overall sign of a sound makes no difference to the ear.

### 8. Cleaning up the signal

**The high-pass filter.** Slow drift of the object or the camera shows up as a large, slow wander in the signal, far below any audible frequency. A high-pass filter removes it. We use a 4th-order Butterworth filter. With cutoff $`f_c`$ its power response is

```math
\lvert H(f) \rvert^{2} = \frac{1}{1 + \left( f_c / f \right)^{8}} .
```

It is applied once forward and once backward in time (`scipy.signal.sosfiltfilt`). That cancels the filter's delay, so the sound isn't shifted in time, and squares its response. The default cutoff follows the paper's rule, $`1/20`$ of the Nyquist frequency, kept within $`20`$ to $`100\,\text{Hz}`$:

```math
f_c = \min\!\left( \max\!\left( \frac{f_s}{40},\ 20\,\text{Hz} \right),\ 100\,\text{Hz} \right).
```

At $`f_s = 2200\,\text{Hz}`$ this gives $`f_c = 55\,\text{Hz}`$. The code applies the filter to all 18 band signals, before they are combined.

**Spectral subtraction (`--denoise`).** Some noise doesn't drift; it hums. In the Chips2 video the signal has steady lines at $`120`$, $`240`$, $`300`$ and $`480\,\text{Hz}`$, all multiples of $`60\,\text{Hz}`$, the mains frequency. The most likely source is the lights flickering. Spectral subtraction [[13]](#ref-boll) removes noise like that.

First cut the signal into short overlapping pieces and take the spectrum of each. This is the short-time Fourier transform $`X(m, f)`$, where $`m`$ numbers the pieces; we use pieces of $`256`$ samples. Then ask, at each frequency, what level is always there. Music comes and goes and a hum doesn't, so the median over all the pieces is a good estimate of the noise power:

```math
\hat N(f) = \operatorname*{median}_{m} \; \lvert X(m, f) \rvert^{2} .
```

Next, scale each piece of the spectrum down according to how much of it is noise,

```math
G(m, f) = \sqrt{ \max\!\left( 1 - \alpha\, \frac{\hat N(f)}{\lvert X(m, f) \rvert^{2}},\ \beta^{2} \right) },
\qquad
\hat S(m, f) = G(m, f)\, X(m, f),
```

and turn $`\hat S`$ back into a signal with the inverse transform. Where the signal is far above the noise, $`G \approx 1`$ and nothing changes. Where it is all noise, $`G`$ drops to the floor $`\beta`$. We use $`\alpha = 2`$, which removes a little more than the estimated noise, and $`\beta = 0.05`$, which turns noisy parts down by $`26\,\text{dB}`$ instead of cutting them to silence, so the result doesn't sound chopped. The phase of $`X`$ is kept as it was.

### 9. Writing the sound file

The signal is stretched to fill the range $`[-1, 1]`$,

```math
y(t) = \frac{2\, s(t) - \left( s_{\max} + s_{\min} \right)}{s_{\max} - s_{\min}} ,
```

and saved as a 16-bit WAV file, $`\operatorname{round}\!\left(32767\, y(t)\right)`$. Each video frame becomes one audio sample, so the sample rate is the frame rate $`f_s`$, rounded to a whole number.

That sets the highest frequency the sound can contain. A sampled signal can only hold frequencies up to half its sample rate, the **Nyquist frequency**:

```math
f_{\max} = \frac{f_s}{2} .
```

At $`2200`$ frames per second, $`f_{\max} = 1100\,\text{Hz}`$. Middle C on a piano is $`262\,\text{Hz}`$, and the fundamental of most voices is below $`300\,\text{Hz}`$, so melodies and the pitch of speech come through. The higher frequencies that make consonants clear do not.

### The whole pipeline

1. Read each frame, convert it to grey, and crop it to the ROI if there is one.
2. Take its 3-level DTCWT: 18 complex sub-bands.
3. For every coefficient, compute the phase change since the previous frame ([section 5](#5-measuring-motion-from-one-frame-to-the-next)).
4. For every band, add up the phase changes weighted by $`A^2`$ ([section 6](#6-many-pixels-one-number-per-band)), then sum over time.
5. High-pass each band signal ([section 8](#8-cleaning-up-the-signal)).
6. Combine the bands into $`d_x`$ and $`d_y`$, find the main direction, and project onto it ([section 7](#7-six-orientations-one-direction)).
7. Optionally, remove steady noise by spectral subtraction ([section 8](#8-cleaning-up-the-signal)).
8. Scale to $`[-1, 1]`$ and save as a WAV file at $`f_s`$ ([section 9](#9-writing-the-sound-file)).

### Notation

| Symbol | Meaning |
|---|---|
| $`f_s`$ | frame rate of the video, which is also the audio sample rate, in Hz |
| $`t`$ | frame number, $`t = 0, 1, \ldots, N-1`$ |
| $`\ell,\ k`$ | DTCWT level and orientation |
| $`\mathbf{p}`$ | position within a sub-band |
| $`c_{\ell,k,t}(\mathbf{p}) = A\,e^{i\phi}`$ | complex wavelet coefficient, with amplitude $`A`$ and phase $`\phi`$ |
| $`\Delta\phi_{\ell,k,t}(\mathbf{p})`$ | phase change of one coefficient from frame $`t-1`$ to frame $`t`$ |
| $`\Phi_{\ell,k}(t)`$ | amplitude-weighted motion signal of one band |
| $`\theta_k`$ | direction that orientation $`k`$ is sensitive to |
| $`d_x(t),\ d_y(t)`$ | horizontal and vertical motion, combined from all bands |
| $`s(t)`$ | motion along the main vibration direction, before scaling |
| $`f_c`$ | cutoff of the high-pass filter, in Hz |

---

## The original method, and what we changed

Davis et al. built their pipeline from the same ingredients in a slightly different order. In the notation above, with a complex steerable pyramid in place of the DTCWT:

1. Phase against a fixed reference frame $`t_0`$, usually the first: $`\phi_v(t) = \phi(t) - \phi(t_0)`$.
2. One signal per band, $`\Phi_i(t) = \sum_{\mathbf{p}} A^2\, \phi_v(t)`$. This is the same weighting as ours.
3. Shift each band in time by the $`\tau_i`$ that best lines it up with a reference band, then add them: $`\hat s(t) = \sum_i \Phi_i(t - \tau_i)`$.
4. Scale to $`[-1, 1]`$, high-pass at $`20`$ to $`100\,\text{Hz}`$, and denoise: spectral subtraction for accuracy, or a speech enhancement method (Loizou 2005) for intelligibility.

They filmed with a Phantom V10 high-speed camera at $`2`$ to $`20\,\text{kHz}`$, used a pyramid with 4 scales and 2 orientations, and report 2 to 3 hours of MATLAB processing per video.

Earlier versions of this project followed the paper closely. Testing against the sound that was really played showed where that went wrong:

| Step | Original | This project | Why we changed it |
|---|---|---|---|
| Phase | against frame $`0`$ | frame to frame, then summed | Drift makes the phase against frame $`0`$ jump by $`2\pi`$. Synthetic test with $`10`$ pixels of drift: $`-20\,\text{dB}`$ before, $`15.9\,\text{dB}`$ after |
| Combining bands | shift each band to match a reference, then add | rebuild $`(d_x, d_y)`$ and project onto the main direction | On Chips2 the shifts came out as long as the whole clip and cost $`7\,\text{dB}`$. A deforming bag's bands are not time-shifted copies of each other |
| High-pass | always | on by default, same rule | Without it, $`0.3`$ pixels of drift drowned the sound |
| Denoising | spectral subtraction or speech enhancement | spectral subtraction with a median noise estimate (`--denoise`) | The music starts with the first frame, so there is no quiet stretch to measure the noise in |

The paper also shows how to get sound from an ordinary 60 fps camera. Most cameras have a **rolling shutter**: the sensor reads its rows one after another, so each row is a picture of a slightly different moment and can give one audio sample. With a Pentax K-01 recording 720 rows at 60 fps, the paper reports an effective rate of $`61{,}920`$ samples per second, with a row delay of $`16\,\mu\text{s}`$. About $`30\%`$ of the samples are missing, because the sensor pauses for $`5\,\text{ms}`$ between frames, and the paper fills the gaps by interpolation. The exposure time sets the real limit: a row exposed for $`1/2000\,\text{s}`$ blurs anything faster than about $`2000\,\text{Hz}`$. This project does not implement the rolling shutter method.

---

## Inside the code

Everything is in `visualmic.py`. It has two ways to compute the band signals, and they agree: on a synthetic clip with 4 pixels of drift their outputs correlate at $`1.0`$, and on Chips2 they score the same to two decimals.

**The CPU path** (`extract_audio`) uses the `dtcwt` library.

- The main process reads frames in order and hands them to worker processes (`--jobs`) in blocks of 32.
- Each block also carries the last frame of the block before it, because every frame is compared with the one before it.
- At most two blocks per worker are in memory at a time, so a video of any length fits.
- Frames are converted to float32 first. `dtcwt` then works in float32 instead of float64, which is $`2.3`$ times faster, and the coefficients change by about one part in a million. The sums over positions are still done in float64.

On a 6-core laptop, 4 workers process about 99 frames per second, and 6 workers are slower again at 87. The transform moves a lot of data, so memory bandwidth is the limit, which is why `--jobs` stops at 4 by default.

**The PyTorch path** (`extract_audio_gpu`, `--gpu`) uses [`pytorch_wavelets`](https://github.com/fbcotter/pytorch_wavelets).

- Frames are grouped into batches of `--batch-size`, and the last, smaller batch is processed too.
- Each batch goes through `DTCWTForward` in one call, inside `torch.inference_mode()`.
- Within a batch, each frame is compared with the frame before it. The last frame of each batch stays on the device for the first frame of the next.
- Only the 18 numbers per frame are copied back to the CPU.

`pytorch_wavelets` stores each coefficient as a real and an imaginary part, so the conjugate product of [section 5](#5-measuring-motion-from-one-frame-to-the-next) is written out by hand. For the current coefficient $`c = a + ib`$ and the previous one $`c' = a' + ib'`$:

```math
c\,\overline{c'} = (a a' + b b') + i\,(b a' - a b'),
\qquad
\Delta\phi = \operatorname{atan2}\!\left( b a' - a b',\ a a' + b b' \right).
```

Before it starts, the PyTorch path estimates the GPU memory it needs: about 15 times the size of a float32 frame for each frame in the batch, plus $`300\,\text{MB}`$. For Chips2 at batch size 32 that is $`0.8\,\text{GB}`$. If the estimate is more than 70% of the free memory, it warns you. If the GPU runs out of memory anyway, it stops with a message instead of a traceback.

Both paths read frames until the video ends. They don't trust the frame count stored in the file, which is often wrong, and they warn when it is.

---

## Results

### How the output is scored

`scripts/eval_audio.py` compares a recovered sound $`r`$ with the sound that was played, $`p`$. Both are high-passed at $`50\,\text{Hz}`$ first. The video and the audio recording don't start at the same moment, and the sign of the recovered sound is arbitrary, so the script first finds the delay and sign that line them up best (the peak of the absolute value of their cross-correlation). Then it reports three numbers.

**SNR.** Find the multiple of the played sound that best matches the recovered one, $`a = \langle r, p \rangle / \langle p, p \rangle`$, and count everything else as noise:

```math
\text{SNR} = 10 \log_{10} \frac{\lVert a\,p \rVert^{2}}{\lVert r - a\,p \rVert^{2}} \ \text{dB} .
```

**Segmental SNR.** The same, computed separately on pieces of $`30\,\text{ms}`$. Each piece's value is clamped to $`[-10, 35]\,\text{dB}`$, then the values are averaged [[14]](#ref-hansen). This shows how good the sound is most of the time, rather than letting the loudest passages decide.

**Coherence.** SNR is harsh in one way: the object filters the sound. A bag of chips moves more for low notes than for high ones, so even a perfect measurement of its motion is not an exact copy of the sound. Coherence ignores any such filtering. At each frequency $`f`$ it measures how consistently $`r`$ follows $`p`$:

```math
C_{rp}(f) = \frac{\lvert P_{rp}(f) \rvert^{2}}{P_{rr}(f)\, P_{pp}(f)} ,
```

where $`P_{rp}`$ is the cross-spectrum of the two sounds and $`P_{rr}`$, $`P_{pp}`$ are their power spectra. It is $`1`$ when $`r`$ is a filtered copy of $`p`$, and $`0`$ when they are unrelated. The script averages it over $`100`$ to $`1000\,\text{Hz}`$, weighted by the power of the played sound, so frequencies where nothing was played don't count. Of the three numbers, it is the fairest.

### Scores

Both MIT videos at $`2200\,\text{fps}`$, run on the GPU. Each cell is SNR / segmental SNR (in dB) / coherence:

| Output | Chips2 (bag of chips) | Plant (leaves) |
|---|---|---|
| v2.0.0, default settings | −11.1 / −9.2 / 0.49 | −26.5 / −9.9 / 0.40 |
| current, default settings | −2.9 / −1.5 / 0.49 | −12.5 / −9.9 / 0.45 |
| current, `--denoise` | −2.9 / −1.1 / 0.49 | −5.3 / −4.0 / 0.46 |
| MIT `recovered.wav` (denoised) | −4.0 / −1.1 / 0.60 | −4.7 / −4.0 / 0.46 |

On the plant, the current version with `--denoise` matches MIT's published result on segmental SNR and coherence, and is $`0.6\,\text{dB}`$ behind on SNR. On the bag of chips it scores higher on SNR, the same on segmental SNR, and lower on coherence ($`0.49`$ against $`0.60`$). That last gap comes from the motion measurement itself, before any denoising. It is the main open question.

### Speed

Chips2 (704x400, 38,083 frames) on a laptop with an RTX 4050 (6 GB) and a 6-core Ryzen 7 7445HS:

| Configuration | Time |
|---|---|
| GPU, `--batch-size 32` | 3m 50s |
| GPU, `--roi 100,50,400,300` | 2m 46s |
| CPU, default (4 workers, float32) | 5m 45s |
| CPU, v3.1.0 (one process, float64) | 24m 23s |

---

## Development

### Running the tests

All tests run inside Docker, so you need no local Python dependencies:

```bash
./test.sh              # CPU image: lint and tests
./test.sh gpu          # GPU image: lint and tests (needs the NVIDIA Container Toolkit)
./test.sh --build      # rebuild the image first
```

The tests are in four files:

- **`tests/test_visualmic.py`**: the building blocks. Among other things:
  - the filter must cut an out-of-band tone by more than $`40\,\text{dB}`$;
  - spectral subtraction must remove a steady hum and keep a tone that comes and goes;
  - parallel and single-process output must be identical;
  - bad command-line input must fail early, with the error on stderr.
- **`tests/test_recovery.py`**: recovery of known motion. A texture is moved by exact sub-pixel amounts following a $`100`$ to $`1000\,\text{Hz}`$ chirp, the test signal of the paper's Fig. 5, and the output is scored against it. The cases cover:
  - horizontal, vertical and diagonal motion at the sizes the paper measured;
  - up to $`10`$ pixels of drift;
  - wrong frame counts in the file;
  - the paper's rule that SNR rises by $`6\,\text{dB}`$ when the motion doubles.
- **`tests/test_visualmic_gpu.py`**: the PyTorch path. These tests run on CUDA when there is a GPU and on the CPU otherwise. They also check that the PyTorch and NumPy paths agree.
- **`tests/test_eval_audio.py`**: checks that the scoring script finds a known delay, sign and SNR.

CI runs on every push. One job builds the CPU image and runs lint and the tests. A second job runs the whole suite, PyTorch path included, on a CPU-only PyTorch build. The multi-GB GPU image is built only when its own files change.

### Dependencies

`requirements*.txt` hold the allowed version ranges. The Docker images install from the hash-locked `requirements.lock` and `requirements-gpu.lock`. The base image is pinned by digest, and the GitHub Actions by commit. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to regenerate the locks.

Two pins are deliberate:

- **NumPy below 2**, because both `dtcwt` and `pytorch_wavelets` use functions that NumPy 2 removed.
- **PyTorch 2.1.2**, the version `pytorch_wavelets` has been tested with here.

### Versioning

The version lives in `VERSION` and in `__version__` in `visualmic.py`, and a test checks that the two agree. To release:

1. Update `VERSION` and `__version__`.
2. In `CHANGELOG.md`, move the entries from `[Unreleased]` to a new `[X.Y.Z] - YYYY-MM-DD` section.
3. Commit as `Release vX.Y.Z`, tag with `git tag -a vX.Y.Z -m "Release vX.Y.Z"`, and push the commit and the tag.
4. Rebuild the images with `./docker-build.sh` and `./docker-build-gpu.sh`.

### Project structure

```
visualmic.py               the program: CPU and PyTorch paths
scripts/eval_audio.py      scores recovered sound against the played sound
tests/                     test_visualmic.py, test_recovery.py,
                           test_visualmic_gpu.py, test_eval_audio.py
Dockerfile                 CPU image (python:3.11-slim)
Dockerfile.gpu             GPU image (python:3.11-slim with the PyTorch CUDA wheels)
docker-build.sh            builds and tags the CPU image
docker-build-gpu.sh        builds and tags the GPU image
test.sh                    lint and tests in Docker
requirements*.txt          allowed dependency ranges
requirements*.lock         hash-locked dependencies used by the images
docs/design/               design records
VERSION, CHANGELOG.md, CONTRIBUTING.md
```

---

## Related work

| Year | Work | What it added |
|---|---|---|
| 2012 | Eulerian Video Magnification [[7]](#ref-evm) | Amplifies tiny changes in video so you can see them. The starting point for this line of work |
| 2013 | Phase-Based Video Motion Processing [[2]](#ref-phase) | Showed that the phase of complex steerable pyramid coefficients measures and magnifies small motions |
| 2014 | Riesz Pyramids [[3]](#ref-riesz) | A phase representation less redundant than even the smallest complex steerable pyramid, fast enough for real time |
| 2014 | **The Visual Microphone** [[1]](#ref-vm) | Recovered sound from high-speed video of everyday objects |
| 2016 | Visual Vibration Analysis, Abe Davis's PhD thesis [[5]](#ref-thesis) | Extended the method to vibration modes and material properties |
| 2017 | Local Visual Microphones [[6]](#ref-local) | Combines local vibration signals instead of one global average. Two to three orders of magnitude faster, real time on 20 kHz video, and estimates where the sound came from |
| 2020 | Frame-wise image denoising [[8]](#ref-choong20) | Denoising each frame before recovery raised SNR and intelligibility |
| 2022 | Effect of video resolution [[9]](#ref-choong22) | Upscaling or downscaling the video before recovery did not help, and usually hurt |
| 2023 | Event-based visual microphone [[10]](#ref-event) | Uses an event camera instead of a high-speed camera |
| 2024 | PSO-CNN hybrid [[11]](#ref-pso) | Combines a neural network with particle swarm optimisation to improve the recovered sound |
| 2025 | Single-pixel visual microphone [[12]](#ref-single) | Records the vibration with a single-pixel detector and a spatial light modulator instead of a camera |

Other implementations:

| Implementation | Method | Language |
|---|---|---|
| MIT original | complex steerable pyramid | MATLAB |
| [dsforza96/visual-mic](https://github.com/dsforza96/visual-mic) | complex steerable pyramid (`pyrtools`) | Python |
| This repository | 2D DTCWT (`dtcwt`, `pytorch_wavelets`) | Python |

---

## References

1. <a id="ref-vm"></a>A. Davis, M. Rubinstein, N. Wadhwa, G. J. Mysore, F. Durand, W. T. Freeman. *The Visual Microphone: Passive Recovery of Sound from Video.* ACM Transactions on Graphics 33(4), SIGGRAPH 2014. [Paper](https://people.csail.mit.edu/mrub/papers/VisualMic_SIGGRAPH2014.pdf), [project page](https://people.csail.mit.edu/mrub/VisualMic/), [data](https://data.csail.mit.edu/vidmag/VisualMic/).
2. <a id="ref-phase"></a>N. Wadhwa, M. Rubinstein, F. Durand, W. T. Freeman. *Phase-Based Video Motion Processing.* ACM Transactions on Graphics 32(4), SIGGRAPH 2013. [Project page](https://people.csail.mit.edu/nwadhwa/phase-video/).
3. <a id="ref-riesz"></a>N. Wadhwa, M. Rubinstein, F. Durand, W. T. Freeman. *Riesz Pyramids for Fast Phase-Based Video Magnification.* IEEE International Conference on Computational Photography (ICCP), 2014. [Project page](https://people.csail.mit.edu/nwadhwa/riesz-pyramid/).
4. <a id="ref-dtcwt"></a>I. W. Selesnick, R. G. Baraniuk, N. G. Kingsbury. *The Dual-Tree Complex Wavelet Transform.* IEEE Signal Processing Magazine 22(6), 123-151, 2005. [doi:10.1109/MSP.2005.1550194](https://doi.org/10.1109/MSP.2005.1550194).
5. <a id="ref-thesis"></a>A. Davis. *Visual Vibration Analysis.* PhD thesis, MIT, 2016. [Thesis](https://abedavis.com/files/papers/thesis.pdf).
6. <a id="ref-local"></a>M. A. Shabani, L. Samadfam, M. A. Sadeghi. *Local Visual Microphones: Improved Sound Extraction from Silent Video.* British Machine Vision Conference (BMVC), 2017. [arXiv:1801.09436](https://arxiv.org/abs/1801.09436).
7. <a id="ref-evm"></a>H.-Y. Wu, M. Rubinstein, E. Shih, J. Guttag, F. Durand, W. T. Freeman. *Eulerian Video Magnification for Revealing Subtle Changes in the World.* ACM Transactions on Graphics 31(4), SIGGRAPH 2012. [Project page](https://people.csail.mit.edu/mrub/evm/).
8. <a id="ref-choong20"></a>R.-J. Choong, W.-S. Yap, Y. C. Hum, Y. K. Tee. *Improving the Quality of Sound Recovered Using the Visual Microphone with Frame-wise Image Denoising Preprocessing.* Journal of Physics: Conference Series 1627, 012024, 2020. [doi:10.1088/1742-6596/1627/1/012024](https://doi.org/10.1088/1742-6596/1627/1/012024).
9. <a id="ref-choong22"></a>R.-J. Choong, W.-S. Yap, Y. C. Hum, Y. K. Tee. *A Study on the Effect of Video Resolution on the Quality of Sound Recovered using the Visual Microphone.* 4th International Conference on Image, Video and Signal Processing (IVSP), 113-117, 2022. [doi:10.1145/3531232.3531248](https://doi.org/10.1145/3531232.3531248).
10. <a id="ref-event"></a>R. Niwa, T. Fushimi, K. Yamamoto, Y. Ochiai. *Live Demonstration: Event-Based Visual Microphone.* CVPR Workshops, 4054-4055, 2023. [Paper](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Niwa_Live_Demonstration_Event-Based_Visual_Microphone_CVPRW_2023_paper.pdf).
11. <a id="ref-pso"></a>K. A. A. Yaqoub, A.-W. S. Ibrahim, A. S. Abdul Jabar. *Retrieving Visual Microphone Sound Using the PSO-CNN Hybrid Technique.* AIP Conference Proceedings 3229, 040007, 2024. [doi:10.1063/5.0237158](https://doi.org/10.1063/5.0237158).
12. <a id="ref-single"></a>W. Zhang, C. Shao, H. Fan, Y. Wang, S. Li, X. Yao. *A Visual Microphone Based on Computational Imaging.* Optics Express 33, 33505-33514, 2025. [doi:10.1364/OE.565525](https://doi.org/10.1364/OE.565525).
13. <a id="ref-boll"></a>S. F. Boll. *Suppression of Acoustic Noise in Speech Using Spectral Subtraction.* IEEE Transactions on Acoustics, Speech, and Signal Processing 27(2), 113-120, 1979.
14. <a id="ref-hansen"></a>J. H. L. Hansen, B. L. Pellom. *An Effective Quality Evaluation Protocol for Speech Enhancement Algorithms.* International Conference on Spoken Language Processing (ICSLP), 1998.

Software: [`dtcwt`](https://github.com/rjw57/dtcwt) ([documentation](https://dtcwt.readthedocs.io/)) and [`pytorch_wavelets`](https://github.com/fbcotter/pytorch_wavelets).

---

## Follow me
<a href="https://x.com/joelk1jose" target="_blank"><img src=".github/images/x.png" width="30"></a>&nbsp;&nbsp;
<a href="https://github.com/joeljose" target="_blank"><img src=".github/images/gthb.png" width="30"></a>&nbsp;&nbsp;
<a href="https://www.linkedin.com/in/joel-jose-527b80102/" target="_blank"><img src=".github/images/lnkdn.png" width="30"></a>

<h3 align="center">If this was useful, star the repository.</h3>
