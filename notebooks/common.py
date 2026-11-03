"""Shared settings for all notebooks.

Every number here is traced to a source in papers/PAPER_NUMBERS.md.
"""
import hashlib
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
RESULTS = ROOT / "results"
for _d in (DATA, FIGS, RESULTS):
    _d.mkdir(exist_ok=True)

GWOSC = "https://gwosc.org/eventapi/html/GWTC-1-confident/GW150914/v3/"

# 32 s files used by the ringdown docs example (MD5s from the docs page)
FILES_32S = {
    "H1": ("H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5", "b345645c0488287301824fe617f58dc2"),
    "L1": ("L-L1_GWOSC_16KHZ_R1-1126259447-32.hdf5", "5eb246ff21364d38471c95452f012853"),
}
# 4096 s files used by Isi & Farr 2022 (MD5s from their released notebook, cell 9)
FILES_4096S = {
    "H1": ("H-H1_GWOSC_16KHZ_R1-1126257415-4096.hdf5", "9a853be0f769fd177d40880dbd56af62"),
    "L1": ("L-L1_GWOSC_16KHZ_R1-1126257415-4096.hdf5", "299f1ca9c089c4802d48f313ce2e6567"),
}

# Isi et al. 2019 reference (peak) time at Hanford, and the same instant at
# the geocentre as used by the ringdown docs / Isi & Farr notebook.
T_REF_H1 = 1126259462.423
T_REF_GEO = 1126259462.4083147
SKY = dict(ra=1.95, dec=-1.27, psi=0.82)

# Time unit of Isi & Farr 2022: t_M = G M / c^3 with M = 69 Msun (= 0.34 ms)
M_REF = 69.0


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(files):
    """Download GWOSC files (if missing) and check their MD5 checksums."""
    paths = {}
    for ifo, (name, expected) in files.items():
        p = DATA / name
        if not p.exists():
            # download to a .part file first, so an interrupted download is
            # never mistaken for a complete file
            print(f"downloading {name} ...")
            part = p.with_suffix(".part")
            urllib.request.urlretrieve(GWOSC + name, part)
            part.rename(p)
        got = md5(p)
        status = "OK" if got == expected else "MISMATCH"
        print(f"{name}: md5 {got} [{status}]")
        if got != expected:
            raise RuntimeError(f"checksum mismatch for {name}")
        paths[ifo] = str(p)
    return paths


def t_M_seconds(M=M_REF):
    """Seconds in one unit of G M / c^3 for a mass M in solar masses."""
    import ringdown as rd
    return M * rd.qnms.T_MSUN


TWO_TONES = [(1, -2, 2, 2, 0), (1, -2, 2, 2, 1)]

# "Paper-like" settings, copied from Isi & Farr 2022's released notebook
# (cells 16, 19, 21, 30) and translated to ringdown 1.1.0 option names:
#   T=0.2 s, 2048 Hz, 20 Hz high-pass, cos(iota) <= -0.99, M in [35, 140],
#   flat prior on the amplitudes A up to 1e-20 (their Fig. 7 caption).
PAPER_PRIORS = dict(a_scale_max=1e-20, m_min=35.0, m_max=140.0,
                    cosi_min=-1.0, cosi_max=-0.99,
                    marginalized=False, flat_amplitude_prior=True)


def make_paper_fit(dt0_M=0.0, modes=TWO_TONES, data_paths=None):
    """Fit object with start time t_ref + dt0_M * t_M (M = 69 Msun), ready to run."""
    import ringdown as rd
    if data_paths is None:
        data_paths = fetch(FILES_4096S)
    fit = rd.Fit(modes=modes)
    fit.set_target(T_REF_GEO + dt0_M * t_M_seconds(), duration=0.2, **SKY)
    fit.load_data(data_paths, ifos=["H1", "L1"], kind="GWOSC")
    fit.condition_data(ds=8, f_min=20)
    fit.compute_acfs(f_min=20)
    fit.update_model(**PAPER_PRIORS)
    return fit


def significance(a1):
    """Overtone significance: mean/std (Isi+ 2019) and median/std (Isi & Farr 2022)."""
    import numpy as np
    a1 = np.asarray(a1).ravel()
    return a1.mean() / a1.std(), np.median(a1) / a1.std()


def contour90(ax, x, y, color, label, level=0.9):
    """Draw the contour enclosing `level` of the probability of samples (x, y)."""
    import numpy as np
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(np.vstack([x, y]))
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 120),
                         np.linspace(y.min(), y.max(), 120))
    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    # density threshold such that the region above it holds `level` of the samples
    thresh = np.quantile(kde(np.vstack([x, y])), 1 - level)
    ax.contour(xx, yy, z, levels=[thresh], colors=[color])
    ax.plot([], [], c=color, label=label)


def configure_jax(n_chains=4):
    """Same setup as the ringdown docs: CPU, 4 parallel chains, single-threaded numpy."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpyro
    numpyro.set_host_device_count(n_chains)
    numpyro.set_platform("cpu")
    import warnings
    warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
