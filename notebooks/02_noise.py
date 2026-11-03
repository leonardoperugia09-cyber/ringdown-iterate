# %% [markdown]
# # Step 2 – Understand the noise
#
# We estimate the noise **power spectral density** (PSD) of each detector,
# using data *away from the event*: 1024 s that end 32 s before the peak, so
# the signal itself does not contaminate the estimate. We plot the
# **amplitude spectral density** ASD = √PSD, which has units of
# strain/√Hz.
#
# Method: Welch's method. Cut the 1024 s into 4 s pieces, Fourier transform
# each one, square, and average. Averaging many pieces beats down the
# randomness of any single piece.

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import welch, butter, sosfiltfilt
from common import FILES_4096S, FIGS, T_REF_H1, fetch

# White noise from my Build. Set these to the values you used; if left as
# None, the white-noise line is drawn at LIGO's best (lowest) ASD level so the
# *shapes* can be compared.
BUILD_SIGMA = None        # per-sample std of the white noise in the Build
BUILD_FS = None           # sample rate of the Build (Hz)

paths = fetch(FILES_4096S)


def read_gwosc(path):
    with h5py.File(path, "r") as f:
        h = f["strain/Strain"][:]
        t0 = f["meta/GPSstart"][()]
        T = f["meta/Duration"][()]
    fs = len(h) / T
    return t0 + np.arange(len(h)) / fs, h, fs


data = {ifo: read_gwosc(p) for ifo, p in paths.items()}

# %%
OFF_END = T_REF_H1 - 32.0   # stop 32 s before the event
OFF_LEN = 1024.0            # seconds of off-source data
NPERSEG_S = 4.0             # Welch segment length (s)

asd = {}
for ifo, (t, h, fs) in data.items():
    sel = (t >= OFF_END - OFF_LEN) & (t < OFF_END)
    off = h[sel]
    assert np.all(np.isfinite(off)), f"{ifo}: gap (NaN) in off-source data"
    f, psd = welch(off, fs=fs, nperseg=int(NPERSEG_S * fs))
    asd[ifo] = (f, np.sqrt(psd))
    print(f"{ifo}: {sel.sum()/fs:.0f} s off-source, "
          f"min ASD = {np.sqrt(psd[(f>20)&(f<2000)]).min():.2e} /√Hz "
          f"at {f[(f>20)&(f<2000)][np.argmin(psd[(f>20)&(f<2000)])]:.0f} Hz")

# %% [markdown]
# ## Coloured (real) noise vs white (Build) noise
# White noise has the **same power at every frequency**, so its ASD is a flat
# line: for per-sample std σ at sample rate $f_s$ the one-sided ASD is
# $\sigma\sqrt{2/f_s}$.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
for ifo, (f, a) in asd.items():
    ax.loglog(f[1:], a[1:], lw=1, label=f"{ifo} (real, coloured)")
band = (asd["H1"][0] > 20) & (asd["H1"][0] < 2000)
if BUILD_SIGMA is not None:
    white_level = BUILD_SIGMA * np.sqrt(2 / BUILD_FS)
    label = f"white noise from my Build (σ={BUILD_SIGMA:g})"
else:
    white_level = asd["H1"][1][band].min()
    label = "white noise (flat), at LIGO's best level"
ax.axhline(white_level, c="k", ls="--", label=label)
ax.axvspan(150, 400, color="gray", alpha=0.15, label="ringdown band (~150–400 Hz)")
ax.set_xlim(10, 4096)
ax.set_ylim(1e-24, 1e-18)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel(r"ASD (strain/$\sqrt{\rm Hz}$)")
ax.set_title("GW150914 noise: 1024 s of off-source data")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "02_asd_vs_white.png", dpi=150)

# %% [markdown]
# ## Whitening
# Whitening = divide the data, frequency by frequency, by the noise ASD. After
# that, every frequency has the same noise level (the noise becomes "white"),
# so a loud signal at 30–300 Hz is no longer hidden by the huge noise at low
# frequency. We also band-pass 30–350 Hz to remove regions with no signal.

# %%
def whiten(h, fs, f_asd, a_asd):
    n = len(h)
    win = np.hanning(n)  # gentle taper so the FFT edges don't ring
    H = np.fft.rfft(h * win)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    a = np.interp(freqs, f_asd, a_asd)
    return np.fft.irfft(H / a, n) * np.sqrt(2 / fs)  # unit-variance scaling


sos = butter(4, [30, 350], btype="band", fs=16384, output="sos")
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for ifo, (t, h, fs) in data.items():
    sel = np.abs(t - T_REF_H1) < 8  # 16 s around the event
    w = sosfiltfilt(sos, whiten(h[sel], fs, *asd[ifo]))
    tt = t[sel] - T_REF_H1
    z = np.abs(tt) < 0.25
    axs[0].plot(tt[z], h[sel][z] - h[sel][z].mean(), lw=1, label=ifo)
    axs[1].plot(tt[z], w[z], lw=1, label=ifo)
axs[0].set_ylabel("raw strain")
axs[1].set_ylabel("whitened + band-passed\n(arbitrary units)")
axs[1].set_xlabel("time from peak at H1 (s)")
axs[0].set_title("Same 0.5 s of data: raw (top) vs whitened (bottom)")
for ax in axs:
    ax.axvline(0, c="k", ls=":", lw=1)
    ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGS / "02_whitened.png", dpi=150)

# %% [markdown]
# In the whitened data the chirp is visible by eye in both detectors, ending
# near t = 0. The two detectors do not see identical wiggles: L1 receives the
# wave ~7 ms *before* H1 (Isi+ 2019 supplement), and each detector responds
# differently to the two polarizations because of its orientation (Step 3
# explains antenna patterns).
