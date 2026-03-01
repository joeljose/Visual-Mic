# Visual-Mic

When sound hits an object, it causes small vibrations on the object's surface. Here we show how, using only high-speed video of the object, we can extract those minute vibrations and partially recover the sound that produced them, allowing us to turn everyday objects—a glass of water, a potted plant, a box of tissues, or a bag of chips—into visual microphones.

The original project was done by MIT-CSAIL team. They have captured high-speed videos of packet of chips moving due to an audio clip of "Mary Had A Little Lamb" song. The video decomposition was done using a technique called Riesz Pyramids. In our project we use the same videos provided in the MIT-CSAIL website, but we use 2D Dual Tree Complex Wavelet Transform instead.

The videos can be downloaded from [here](http://data.csail.mit.edu/vidmag/VisualMic/)

![](https://github.com/joeljose/assets/raw/master/Visual-Mic/vmic.png)

---

## Table of Contents

- [Setting Up](#setting-up-visual-mic)
- [Part 1: The Original Work (Davis et al., SIGGRAPH 2014)](#part-1-the-original-work-davis-et-al-siggraph-2014)
  - [1.1 The Physical Phenomenon](#11-the-physical-phenomenon)
  - [1.2 Why Not Just Track Pixels?](#12-why-not-just-track-pixels)
  - [1.3 The Key Insight: Phase = Motion](#13-the-key-insight-phase--motion)
  - [1.4 Complex Steerable Pyramid](#14-complex-steerable-pyramid)
  - [1.5 The Original Algorithm](#15-the-original-algorithm--step-by-step)
  - [1.6 Rolling Shutter Trick](#16-rolling-shutter-trick-consumer-cameras)
  - [1.7 Limitations](#17-limitations-of-the-original)
- [Part 2: Our Implementation (2D DTCWT)](#part-2-our-implementation-2d-dtcwt)
  - [2.1 What is the DTCWT?](#21-what-is-the-dual-tree-complex-wavelet-transform)
  - [2.2 DTCWT vs Steerable Pyramid](#22-dtcwt-vs-complex-steerable-pyramid)
  - [2.3 Algorithm Mapped to Code](#23-our-algorithm--mapped-to-code)
  - [2.4 Parameters](#24-parameters-used)
  - [2.5 What Each Scale Captures](#25-what-each-scale-captures)
- [Part 3: Literature Survey](#part-3-literature-survey)
- [Part 4: Denoising](#part-4-denoising)
- [References](#references)

---

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
   pip install -r requirements.txt
   ```
4. Run visualmic.py:
   ```sh
   python visualmic.py -i <input_video>
   python visualmic.py -i testvid.avi -o recovered_audio.wav
   ```
   | Argument | Required | Description |
   |----------|----------|-------------|
   | `-i`, `--input` | Yes | Path to input video file |
   | `-o`, `--output` | No | Output audio path (default: `sound.wav`) |

---

# Part 1: The Original Work (Davis et al., SIGGRAPH 2014)

**Paper:** *"The Visual Microphone: Passive Recovery of Sound from Video"*
**Authors:** Abe Davis, Michael Rubinstein, Neal Wadhwa, Gautham J. Mysore, Frédo Durand, William T. Freeman
**Venue:** ACM Transactions on Graphics (SIGGRAPH 2014), Vol 33, No 4
**Institutions:** MIT CSAIL, Stanford, Adobe Research

## 1.1 The Physical Phenomenon

When sound travels through air, it creates pressure waves. When these waves hit an object's surface, they cause **tiny vibrations** — displacements on the order of micrometers or less. These vibrations are far too small to see with the naked eye, but a high-speed camera recording thousands of frames per second can capture them as subtle pixel-level changes.

**Key insight:** If we can measure those sub-pixel surface displacements over time, we effectively have a recording of the sound pressure wave — we've turned the object into a microphone.

**Example:** Playing "Mary Had A Little Lamb" near a bag of chips causes the bag's surface to vibrate at the frequencies of the music. A high-speed camera (2000–6000 fps) pointed at the bag captures these vibrations as tiny frame-to-frame changes.

## 1.2 Why Not Just Track Pixels?

You might think: "Just compute optical flow between frames and track the motion." The problem:

1. **The motions are sub-pixel** — typically $\frac{1}{100}$ to $\frac{1}{1000}$ of a pixel. Standard optical flow fails at this scale.
2. **Noise dominates** — sensor noise, quantization noise, and lighting fluctuations are all larger than the actual vibration signal.
3. **You need temporal precision** — to recover audio at meaningful frequencies, you need to track motion at every single frame with high temporal fidelity.

**Solution:** Instead of tracking pixels in the spatial domain, work in the **frequency domain** using the **phase** of complex wavelet/pyramid coefficients. Phase is far more sensitive to small motions than amplitude.

## 1.3 The Key Insight: Phase = Motion

Consider a 1D signal shifted by a small displacement $\delta$:

$$f(x) \rightarrow f(x + \delta)$$

In the Fourier domain, a spatial shift becomes a **phase shift**:

$$F(\omega) \rightarrow F(\omega) \cdot e^{i\omega\delta}$$

So if you decompose an image into frequency bands and track how the **phase** of each band changes over time, you're directly measuring **local displacement** at that spatial frequency.

For a band-pass filtered signal at spatial frequency $\omega_0$:

$$\Delta\phi \approx \omega_0 \cdot \delta$$

where $\delta$ is the local displacement. **This is the foundation of the entire method.**

### Why phase is better than amplitude

| Property | Amplitude $A$ | Phase $\phi$ |
|----------|---------------|--------------|
| Physical meaning | "How much texture is here" | "Where exactly is this texture positioned" |
| Response to small motion | Relatively stable | Shifts linearly with displacement |
| Sub-pixel sensitivity | Poor | Excellent — detects fractions of a pixel |

## 1.4 Complex Steerable Pyramid

The original paper uses a **Complex Steerable Pyramid** to decompose each video frame.

### What is a steerable pyramid?

A multi-scale, multi-orientation filter bank that decomposes an image into:
- Multiple **scales** (frequency bands): coarse $\rightarrow$ fine detail
- Multiple **orientations** at each scale: e.g., $0°, 30°, 60°, 90°, 120°, 150°$ (for 6 orientations)
- A **lowpass residual** (the blurry base image)
- A **highpass residual** (the finest details)

### Why "complex"?

Each sub-band produces **complex-valued** coefficients. At each spatial location $(x, y)$, for scale $s$ and orientation $\theta$, you get:

$$C(s, \theta, x, y) = A(s, \theta, x, y) \cdot e^{i \cdot \phi(s, \theta, x, y)}$$

where:
- $A$ = **amplitude** (how strong the texture is at this location/scale/orientation)
- $\phi$ = **phase** (the precise position of the texture pattern)

### Why "steerable"?

The filters can be analytically rotated to any orientation without recomputing — this gives fine directional control and avoids aliasing artifacts.

### Key Properties

- **Translation equivariant:** shifting the input shifts the coefficients predictably (phase changes linearly)
- **Overcomplete** (~21x for 8 orientations): more coefficients than pixels $\rightarrow$ redundancy helps with noise
- **Shift-invariant:** no downsampling artifacts that would corrupt phase measurements

## 1.5 The Original Algorithm — Step by Step

### Input

- High-speed video $V(x, y, t)$ with $N$ frames at $F$ fps
- Grayscale frames (color not needed for vibration)

### Step 1: Decompose Every Frame

For each frame $t = 0, 1, \ldots, N-1$:

$$\{C(s, \theta, x, y, t)\} = \text{ComplexSteerablePyramid}(V(:,:,t))$$

This gives complex coefficients at $S$ scales and $K$ orientations.

### Step 2: Extract Amplitude and Phase

For each coefficient:

$$A(s, \theta, x, y, t) = |C(s, \theta, x, y, t)|$$

$$\phi(s, \theta, x, y, t) = \angle C(s, \theta, x, y, t)$$

### Step 3: Compute Phase Variation (Local Motion Signal)

Choose a reference frame $t_0$ (usually frame 0). For each subsequent frame:

$$\phi_v(s, \theta, x, y, t) = \phi(s, \theta, x, y, t) - \phi(s, \theta, x, y, t_0)$$

This phase difference is proportional to how much the texture at location $(x, y)$ has moved since the reference frame, at that particular scale and orientation.

**Why subtract the reference?** The absolute phase values encode the texture pattern itself (which we don't care about). By subtracting the reference, we isolate the *change* — which is the vibration.

### Step 4: Compute Global Motion Signal (Amplitude-Weighted Spatial Average)

For each scale $s$ and orientation $\theta$, collapse the spatial dimensions:

$$\Phi(s, \theta, t) = \sum_{x,y} A(s, \theta, x, y, t)^2 \cdot \phi_v(s, \theta, x, y, t)$$

**Why weight by $A^2$?**
- Regions with strong texture (high amplitude) give **reliable** phase measurements
- Regions with weak/no texture (low amplitude) have **noisy/random** phase — we want to suppress these
- $A^2$ weighting is effectively a "reliability-weighted average" that emphasizes trustworthy measurements

This produces one 1D time signal per $(s, \theta)$ pair.

### Step 5: Temporal Alignment Across Sub-bands

Different scales and orientations may have phase offsets relative to each other. Align them using cross-correlation:

1. Pick a reference sub-band (e.g., scale 0, orientation 0): $\text{ref} = \Phi(0, 0, t)$
2. For each other $(s, \theta)$, find the time shift that maximizes correlation:

$$\tau(s, \theta) = \arg\max_{\tau} \sum_t \text{ref}(t) \cdot \Phi(s, \theta, t - \tau)$$

3. Shift each sub-band signal by its optimal lag $\tau$.

### Step 6: Average Across Scales and Orientations

$$\hat{s}(t) = \sum_{s, \theta} \Phi(s, \theta, t - \tau(s, \theta))$$

This averaging acts as denoising — the vibration signal is **coherent** across sub-bands (adds constructively) while noise is **incoherent** (partially cancels).

### Step 7: Normalize

$$\hat{s}_{\text{norm}}(t) = \frac{2 \cdot \hat{s}(t) - (\max + \min)}{\max - \min}$$

Maps the signal to $[-1, 1]$ range.

### Step 8: Output Audio

Write as WAV file with **sampling rate = video FPS**.

> **Critical:** If the video is 2200 fps, the audio is sampled at 2200 Hz. By the Nyquist theorem, this captures frequencies up to 1100 Hz — covering most speech fundamental frequencies and low musical tones.

## 1.6 Rolling Shutter Trick (Consumer Cameras)

High-speed cameras are expensive. But most consumer cameras have **rolling shutter** — the sensor reads rows sequentially, not all at once. Each row is exposed at a slightly different time.

For a 60 fps camera with 480 rows:
- Each row is a separate temporal sample
- Effective sampling rate: $60 \times 480 / 60 \approx 480$ Hz (8x boost)
- Sufficient to capture speech fundamentals

The algorithm adapts by:
1. Treating each row as a separate temporal sample
2. Computing 1D transforms along rows instead of 2D pyramids
3. Stitching the temporal information together

This allowed recovering intelligible speech from a **standard 60 fps consumer camera**.

## 1.7 Limitations of the Original

1. **Requires high-speed video** for good quality (2000+ fps ideal; 60 fps with rolling shutter is limited)
2. **Object must have visible texture** — smooth featureless surfaces give poor results
3. **Sound-to-noise ratio** depends on object material, distance, and sound volume
4. **Computationally expensive** — complex steerable pyramids are ~21x overcomplete
5. **Global averaging loses spatial information** — all vibrations are mixed together

---

# Part 2: Our Implementation (2D DTCWT)

## 2.1 What is the Dual-Tree Complex Wavelet Transform?

The DTCWT was developed by **Nick Kingsbury** (Cambridge, late 1990s) as an improvement over the standard Discrete Wavelet Transform (DWT).

### The problem with standard DWT

- **Not shift-invariant:** shifting input by 1 pixel completely changes the coefficients
- **Poor directional selectivity:** only separates horizontal, vertical, diagonal — no fine orientations
- **Oscillating coefficients:** makes phase extraction unreliable

### How DTCWT works

Run **two parallel DWT filter banks** (two "trees"):

- **Tree $a$:** uses one set of filters $\rightarrow$ produces real part
- **Tree $b$:** uses a slightly different (quarter-sample shifted) set of filters $\rightarrow$ produces imaginary part

The filters are designed so that Tree $b$'s wavelet is approximately the **Hilbert transform** of Tree $a$'s wavelet. Combining them gives:

$$\psi_{\text{complex}}(x) = \psi_a(x) + i \cdot \psi_b(x)$$

This complex wavelet is **approximately analytic** (has energy only on one side of the frequency spectrum), which provides:

- **Approximate shift invariance** (2x oversampling eliminates most aliasing)
- **Clean phase information** (no oscillation artifacts)

### 2D DTCWT Specifically

For 2D images, the DTCWT produces **6 complex sub-bands per scale**, oriented at approximately:

$$\pm 15°, \quad \pm 45°, \quad \pm 75°$$

This is fewer orientations than a typical steerable pyramid (which might use 8+), but:
- Only **~4x overcomplete** (vs ~21x for steerable pyramid)
- Much faster to compute
- Still provides good directional selectivity
- Phase information is reliable for motion estimation

## 2.2 DTCWT vs Complex Steerable Pyramid

| Property | Complex Steerable Pyramid | 2D DTCWT |
|----------|--------------------------|----------|
| Shift invariant | Yes (exactly) | Approximately |
| Orientations per scale | Configurable (typically 8) | 6 (fixed) |
| Overcompleteness | ~21x (8 orientations) | ~4x |
| Computation speed | Slow (frequency domain) | Fast (filter banks) |
| Phase quality | Excellent | Good |
| Python library | `pyrtools` | `dtcwt` |
| Reconstruction | Perfect | Near-perfect |

**Trade-off:** DTCWT is ~5x more computationally efficient at the cost of slightly fewer orientations and approximate (rather than exact) shift invariance. For the visual microphone application, this is a favorable trade-off — the phase information is still good enough to detect sub-pixel vibrations.

## 2.3 Our Algorithm — Mapped to Code

Here's how `visualmic.py` implements the pipeline, with line references.

### Steps 1–3: Stream Video, DTCWT, and Phase Extraction (lines 22–65)

Frames are streamed directly from the video file — each frame is read, transformed, and discarded immediately, so only one raw frame is in memory at a time. This enables processing of arbitrarily long videos without running out of memory.

```python
def soundfromvid(cap, frameCount, nlevels, orient, ref_no, ref_orient, ref_level):
    tr = dtcwt.Transform2d()
    ref_frame = None
    data = []

    for fc in range(frameCount):
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            break
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        dtcwt_frame = tr.forward(gray, nlevels=nlevels)

        if fc == ref_no:
            ref_frame = dtcwt_frame

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

    data = np.array(data)  # shape: (frameCount, nlevels, orient)
```

`tr.forward()` returns a `Pyramid` object:
- `pyramid.highpasses[level]` has shape $(H_{\text{level}}, W_{\text{level}}, 6)$
- Each value is a **complex number** encoding amplitude and phase

Vectorized NumPy operations on entire 2D spatial slices:

| Operation | Code | Corresponds to |
|-----------|------|----------------|
| Extract amplitude $A$ and phase $\phi$ | `np.abs(...)`, `np.angle(...)` | Step 2 of original |
| Phase variation $\phi_v$ (wrapped to $[-\pi, \pi]$) | `np.angle(np.exp(1j * (phase - ref_phase)))` | Step 3 of original |
| $A^2$-weighted accumulation | `amp*amp * (...)` | Step 4 of original |
| Spatial sum $\sum_{x,y}$ | `np.sum(...)` | Step 4 of original |

Phase wrapping ensures the difference always reflects the true small angular displacement, even when phases cross the $\pm\pi$ boundary.

**Result:** `data[fc, level, angle]` $= \Phi(\text{level}, \text{angle}, fc)$ — one scalar per frame per sub-band.

### Step 4: Temporal Alignment via Cross-Correlation (lines 66–70)

```python
ref_vector = data[:, ref_level, ref_orient].reshape(-1)
for i in range(nlevels):
    for j in range(orient):
        shift_matrix[i,j] = maxTime(ref_vector, data[:,i,j].reshape(-1))
```

The `maxTime` function (lines 10–12) uses `scipy.signal.correlate` for $O(n \log n)$ cross-correlation:

```python
def maxTime(a, b):
    c = signal.correlate(a, b, mode='full')
    return np.argmax(c) - (len(b) - 1)
```

### Step 5: Sum Across Sub-bands with Temporal Shifts (lines 72–76)

```python
for fc in range(frameCount):
    for i in range(nlevels):
        for j in range(orient):
            sound_raw[fc] += data[fc - int(shift_matrix[i,j]), i, j]
```

### Step 6: Normalize to $[-1, 1]$ (lines 77–83)

```python
p_min = np.min(sound_raw)
p_max = np.max(sound_raw)
if p_max == p_min:
    sound_data = np.zeros_like(sound_raw)  # silent output if no motion
else:
    sound_data = ((2 * sound_raw) - (p_min + p_max)) / (p_max - p_min)
```

Includes a guard against division by zero when no motion is detected.

### Step 7: Output WAV (lines 14–20, called at line 132)

```python
def npTowav(np_file, output_name, sps):
    waveform_integers = np.int16(np_file * 32767)
    write(output_name, sps, waveform_integers)
```

The sampling rate `sps` is set to the video's FPS, ensuring the output audio matches the temporal resolution of the input video.

## 2.4 Parameters Used

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `nlevels` | 3 | Number of wavelet decomposition scales |
| `orient` | 6 | Number of orientations per scale (fixed by DTCWT) |
| `ref_no` | 0 | Reference frame index (first frame) |
| `ref_level` | 0 | Reference sub-band: finest scale |
| `ref_orient` | 0 | Reference sub-band: first orientation (~$+15°$) |

## 2.5 What Each Scale Captures

With 3 levels of DTCWT decomposition:

| Level | Spatial Frequency | What It Captures | Spatial Resolution |
|-------|-------------------|------------------|--------------------|
| 0 (finest) | High | Fine textures, edges, sharp details | $H/2 \times W/2$ |
| 1 (middle) | Medium | Medium-scale patterns | $H/4 \times W/4$ |
| 2 (coarsest) | Low | Broad structures, large features | $H/8 \times W/8$ |

The vibration signal is present across all scales (the whole surface moves), but the **signal-to-noise ratio** varies:
- **Fine scales:** more spatial locations to average $\rightarrow$ better denoising
- **Coarse scales:** fewer locations but stronger phase response to motion

---

# Part 3: Literature Survey

## Foundational Work

| Year | Paper | Key Contribution |
|------|-------|------------------|
| 2012 | Eulerian Video Magnification (SIGGRAPH) | Predecessor: amplifies color/intensity changes to visualize motion |
| 2013 | Phase-Based Video Motion Processing (SIGGRAPH) | Established that phase of complex steerable pyramid coefficients = local motion |
| 2014 | Riesz Pyramids (ICCP) | Compact 4x overcomplete pyramid for real-time phase-based processing |
| **2014** | **The Visual Microphone (SIGGRAPH)** | **Recovered sound from video using phase-based motion analysis** |
| 2016 | Visual Vibration Analysis (PhD Thesis, Abe Davis) | Extended to modal analysis, material properties, damping estimation |

## Follow-Up Research

| Year | Work | Advance |
|------|------|---------|
| 2018 | Local Visual Microphones | Local vibration aggregation (not global averaging), 100–1000x speedup, sound direction estimation |
| 2022 | Effect of Video Resolution | Studies resolution impact on recovery quality; frame-wise denoising preprocessing |
| 2023 | Event-Based Visual Microphone (CVPR Workshop) | Neuromorphic event cameras for cheap, efficient vibration capture |
| 2024 | PSO-CNN Hybrid | Particle Swarm Optimization + CNN for enhanced sound restoration |
| 2025 | Single-Pixel Visual Microphone (Optica) | Single-pixel imaging with spatial light modulator — no expensive high-speed camera needed |

## Alternative Implementations

| Implementation | Technique | Language |
|----------------|-----------|----------|
| MIT Original | Complex Steerable Pyramid | MATLAB |
| [dsforza96/visual-mic](https://github.com/dsforza96/visual-mic) | Complex Steerable Pyramid (`pyrtools`) | Python |
| **This repo (Visual-Mic)** | **2D DTCWT (`dtcwt`)** | **Python** |

---

# Part 4: Denoising

Denoising is a separate post-processing step. We apply image-based morphological filtering to the audio spectrograms, and then reconstruct audio from the processed spectrogram. Since denoising involves multiple stages, it is maintained as a separate project: [audio_denoising](https://github.com/joeljose/audio_denoising)

---

# References

1. Davis, A., Rubinstein, M., Wadhwa, N., Mysore, G.J., Durand, F., & Freeman, W.T. (2014). *The Visual Microphone: Passive Recovery of Sound from Video.* ACM Transactions on Graphics (SIGGRAPH), 33(4).
   [Paper PDF](https://people.csail.mit.edu/mrub/papers/VisualMic_SIGGRAPH2014.pdf) | [Project Page](https://people.csail.mit.edu/mrub/VisualMic/)

2. Wadhwa, N., Rubinstein, M., Durand, F., & Freeman, W.T. (2013). *Phase-Based Video Motion Processing.* ACM Transactions on Graphics (SIGGRAPH).
   [Project Page](https://people.csail.mit.edu/nwadhwa/phase-video/)

3. Wadhwa, N., Rubinstein, M., Durand, F., & Freeman, W.T. (2014). *Riesz Pyramids for Fast Phase-Based Video Magnification.* IEEE ICCP.
   [Project Page](https://people.csail.mit.edu/nwadhwa/riesz-pyramid/)

4. Selesnick, I.W., Baraniuk, R.G., & Kingsbury, N.G. (2005). *The Dual-Tree Complex Wavelet Transform.* IEEE Signal Processing Magazine, 22(6), 123–151.
   [Tutorial PDF](https://eeweb.engineering.nyu.edu/iselesni/pubs/CWT_Tutorial.pdf)

5. Davis, A. (2016). *Visual Vibration Analysis.* PhD Thesis, MIT.
   [Thesis PDF](https://abedavis.com/files/papers/thesis.pdf)

6. Shen, M. & Bhatt, S. (2018). *Local Visual Microphones: Improved Sound Extraction from Silent Video.*
   [arXiv:1801.09436](https://arxiv.org/abs/1801.09436)

7. Niwa, T., Fushimi, T., Yamamoto, S., & Ochiai, Y. (2023). *Live Demonstration: Event-based Visual Microphone.* CVPR Workshop on Event-based Vision.
   [Paper PDF](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Niwa_Live_Demonstration_Event-Based_Visual_Microphone_CVPRW_2023_paper.pdf)

8. dtcwt Python library.
   [Documentation](https://dtcwt.readthedocs.io/) | [GitHub](https://github.com/rjw57/dtcwt)

9. MIT CSAIL Visual Microphone Dataset.
   [Download](http://data.csail.mit.edu/vidmag/VisualMic/)

---

## Follow Me
<a href="https://twitter.com/joelk1jose" target="_blank"><img class="ai-subscribed-social-icon" src="https://github.com/joeljose/assets/blob/master/images/tw.png" width="30"></a>
<a href="https://github.com/joeljose" target="_blank"><img class="ai-subscribed-social-icon" src="https://github.com/joeljose/assets/blob/master/images/gthb.png" width="30"></a>
<a href="https://www.linkedin.com/in/joel-jose-527b80102/" target="_blank"><img class="ai-subscribed-social-icon" src="https://github.com/joeljose/assets/blob/master/images/lnkdn.png" width="30"></a>

<h3 align="center">Show your support by starring the repository 🙂</h3>
