"""Score recovered audio against the sound that was actually played.

Usage:
    python scripts/eval_audio.py recovered.wav --ref input.wav [--ref-recovered mit.wav ...]

Both signals are high-passed, the recovered one is aligned to the reference
(lag and sign by cross-correlation, since the video and audio clips start at
different times and the recovered sign is arbitrary), then compared over the
overlap with a least-squares gain. Reports SNR and segmental SNR in dB, and
magnitude-squared coherence over 100-1000 Hz (0-1, weighted by the
reference spectrum). Coherence ignores
any linear filtering, so it measures how much of the played sound the pipeline
captured, whatever the object's frequency response (Davis et al. Sec. 5.1
model the object as an LTI system).

The recovered signal is object motion, i.e. the played sound filtered by the
speaker, room and object response, so even a perfect pipeline won't reach a
high SNR. Use the numbers to compare pipeline versions on the same video,
and MIT's own (denoised) recovered.wav as an upper reference.
"""

import argparse

import numpy as np
from scipy import signal
from scipy.io import wavfile

HIGHPASS_HZ = 50
SEG_MS = 30


def load(path):
    rate, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return rate, x


def highpass(x, rate):
    sos = signal.butter(4, HIGHPASS_HZ / (rate / 2), 'high', output='sos')
    return signal.sosfiltfilt(sos, x)


def align(rec, ref):
    """Return (lag, rec_overlap, ref_overlap) with rec[i + lag] ~ ref[i]."""
    c = signal.correlate(rec, ref, mode='full', method='fft')
    lag = int(np.argmax(np.abs(c))) - (len(ref) - 1)
    start, stop = max(0, -lag), min(len(ref), len(rec) - lag)
    if stop - start <= 0:
        raise ValueError("signals do not overlap")
    return lag, rec[start + lag:stop + lag], ref[start:stop]


def scores(rec, ref, rate):
    gain = (rec @ ref) / (ref @ ref)
    target = gain * ref
    noise = rec - target
    snr = 10 * np.log10((target @ target) / (noise @ noise))
    n = max(1, int(rate * SEG_MS / 1000))
    segs = len(ref) // n
    t = target[:segs * n].reshape(segs, n)
    e = noise[:segs * n].reshape(segs, n)
    seg = 10 * np.log10(((t ** 2).sum(1) + 1e-12) / ((e ** 2).sum(1) + 1e-12))
    return snr, float(np.mean(np.clip(seg, -10, 35)))  # clamp as in Hansen & Pellom 1998


def coherence(rec, ref, rate):
    # Weighted by the reference spectrum: music and speech leave most bins
    # empty, and coherence there is just noise.
    f, c = signal.coherence(rec, ref, fs=rate, nperseg=256)
    _, p = signal.welch(ref, fs=rate, nperseg=256)
    band = (f >= 100) & (f <= 1000)
    return float(np.sum(c[band] * p[band]) / np.sum(p[band]))


def evaluate(rec_path, ref_path):
    rate, rec = load(rec_path)
    ref_rate, ref = load(ref_path)
    if ref_rate != rate:
        ref = signal.resample_poly(ref, rate, ref_rate)
    lag, r, f = align(highpass(rec, rate), highpass(ref, rate))
    snr, ssnr = scores(r, f, rate)
    return {'rate': rate, 'lag': lag, 'overlap_s': len(f) / rate, 'snr_db': snr, 'segsnr_db': ssnr,
            'coherence': coherence(r, f, rate)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('recovered', nargs='+', help='Recovered WAV file(s) to score')
    parser.add_argument('--ref', required=True, help='WAV of the sound that was played')
    args = parser.parse_args()
    print(f"{'file':50s} {'lag':>7s} {'overlap':>8s} {'SNR':>8s} {'segSNR':>8s} {'coh':>6s}")
    for path in args.recovered:
        s = evaluate(path, args.ref)
        print(f"{path[-50:]:50s} {s['lag']:7d} {s['overlap_s']:7.1f}s {s['snr_db']:7.1f}  {s['segsnr_db']:7.1f}  {s['coherence']:5.2f}")


if __name__ == '__main__':
    main()
