# %% [markdown]
# # Step 1 – Get the real data
#
# We download the GW150914 strain from GWOSC for LIGO Hanford (H1) and LIGO
# Livingston (L1), check the files are exactly the official ones (MD5
# checksum), and plot the raw strain.
#
# Two sets of files are used in this project:
# * **32 s** files: what the `ringdown` docs example uses (Step 3).
# * **4096 s** files: what Isi & Farr 2022 used (Steps 2, 4, 5). The extra
#   hour of data lets us measure the noise far from the event.

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
from common import FILES_32S, FILES_4096S, FIGS, T_REF_H1, fetch

paths32 = fetch(FILES_32S)
paths4096 = fetch(FILES_4096S)


# %%
def read_gwosc(path):
    """Return (time array, strain array, sample rate) from a GWOSC HDF5 file."""
    with h5py.File(path, "r") as f:
        h = f["strain/Strain"][:]
        t0 = f["meta/GPSstart"][()]
        T = f["meta/Duration"][()]
    fs = len(h) / T
    return t0 + np.arange(len(h)) / fs, h, fs


raw = {ifo: read_gwosc(p) for ifo, p in paths32.items()}
for ifo, (t, h, fs) in raw.items():
    print(f"{ifo}: {len(h)} samples at {fs:.0f} Hz, {len(h)/fs:.0f} s, "
          f"strain std = {np.std(h):.2e}")

# %% [markdown]
# ## Raw strain
# Left: the whole 32 s. Right: zoom on ±0.2 s around the peak time quoted by
# Isi et al. 2019 (GPS 1126259462.423 at H1).

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ifo, (t, h, fs) in raw.items():
    axs[0].plot(t - T_REF_H1, h, lw=0.5, label=ifo)
    sel = np.abs(t - T_REF_H1) < 0.2
    axs[1].plot(t[sel] - T_REF_H1, h[sel], lw=1, label=ifo)
for ax in axs:
    ax.set_xlabel("time from peak (s)")
    ax.set_ylabel("strain $h(t)$")
    ax.legend()
axs[0].set_title("Raw strain, 32 s")
axs[1].set_title("Zoom: ±0.2 s around the peak")
axs[1].axvline(0, c="k", ls=":", lw=1)
fig.tight_layout()
fig.savefig(FIGS / "01_raw_strain.png", dpi=150)

# %% [markdown]
# **Why can't we see the signal?** The raw strain is dominated by very
# low-frequency noise (below ~20 Hz) whose size is the "strain std" printed
# above (≈2.2e-19 for H1, 2.3e-19 for L1), while the ringdown amplitude is
# only a few ×1e-21 (the ringdown docs report $A_{220} \approx 1.8\times10^{-21}$),
# roughly 100 times smaller, and lives at ~250 Hz. The slow wiggle you see is that
# low-frequency noise. In Step 2 we will measure the noise at each frequency
# and "whiten" the data, which makes the chirp pop out.
