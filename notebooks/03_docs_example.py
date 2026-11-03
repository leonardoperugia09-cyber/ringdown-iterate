# %% [markdown]
# # Step 3 – Reproduce the ringdown docs example
#
# Exactly the settings of the official example
# (https://ringdown.readthedocs.io/en/latest/examples/GW150914.html), run
# twice:
# * **220 only**: the fundamental mode.
# * **220 + 221**: fundamental + first overtone.
#
# Both are **Kerr-constrained**: the frequency and damping rate of every mode
# are computed from just two numbers, the remnant mass $M$ and spin $\chi$.
# The amplitudes and phases of each mode are free.

# %%
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from common import configure_jax, contour90, fetch, FILES_32S, FIGS, RESULTS, T_REF_GEO, SKY

configure_jax()
import ringdown as rd

paths = fetch(FILES_32S)
path_pattern = paths["H1"].replace("H-H1", "{i}-{ifo}")

MODELS = {
    "220": [(1, -2, 2, 2, 0)],
    "220+221": [(1, -2, 2, 2, 0), (1, -2, 2, 2, 1)],
}


def setup_docs_fit(modes):
    fit = rd.Fit(modes=modes)
    fit.set_target(T_REF_GEO, duration=0.2, **SKY)
    fit.load_data(path_pattern, ifos=["H1", "L1"], kind="GWOSC")
    fit.condition_data(ds=8, f_min=20)
    fit.compute_acfs(f_min=20)
    fit.update_model(a_scale_max=5e-20, m_min=40.0, m_max=140.0, cosi=-1)
    return fit


fits, results, runtimes = {}, {}, {}
for name, modes in MODELS.items():
    fits[name] = setup_docs_fit(modes)
    nc = RESULTS / f"03_docs_{name}.nc"
    if nc.exists():
        # reuse a finished run (delete the file to sample again)
        results[name] = az.from_netcdf(nc)
        print(f"{name}: loaded {nc.name}")
        continue
    t = time.time()
    fits[name].run()
    runtimes[name] = time.time() - t
    fits[name].result.to_netcdf(nc)
    results[name] = fits[name].result
    print(f"{name}: sampling took {runtimes[name]:.0f} s")
if runtimes:
    json.dump(runtimes, open(RESULTS / "03_runtimes.json", "w"), indent=1)

# %% [markdown]
# ## What the analysis sees
# Start times in each detector: L1 receives the signal ~7 ms before H1. The
# antenna patterns $F_+, F_\times$ say how strongly each detector responds to
# each polarization from this sky position.

# %%
fit = fits["220+221"]
print("start times:", {k: f"{v:.6f}" for k, v in fit.start_times.items()})
dt = fit.start_times["H1"] - fit.start_times["L1"]
print(f"H1 - L1 delay: {dt*1e3:.2f} ms")
print("antenna patterns (Fp, Fc):", fit.antenna_patterns)

# %% [markdown]
# ## Results: mass, spin, amplitudes

# %%
summ = {}
for name, r in results.items():
    s = az.summary(r, var_names=["m", "chi", "a"], hdi_prob=0.68)
    summ[name] = s
    print(f"\n=== {name} ===")
    print(s[["mean", "sd", "hdi_16%", "hdi_84%", "r_hat", "ess_bulk"]])

# %%
fig, ax = plt.subplots(figsize=(6, 5))
for (name, r), c in zip(results.items(), ["C0", "C1"]):
    p = r.posterior
    contour90(ax, p["m"].values.ravel(), p["chi"].values.ravel(), c, name)
ax.plot(68.5, 0.69, "k+", ms=14, mew=2, label="IMR peak (Isi+ 2019 Fig. 1)")
ax.set_xlabel(r"remnant mass $M$ ($M_\odot$, detector frame)")
ax.set_ylabel(r"remnant spin $\chi$")
ax.set_title("90% credible regions (docs settings)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGS / "03_mass_spin.png", dpi=150)

# %%
a = results["220+221"].posterior["a"]
a220 = a.isel(mode=0).values.ravel()
a221 = a.isel(mode=1).values.ravel()
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(a221, bins=60, density=True, alpha=0.7, label="$A_{221}$ (overtone)")
ax.hist(a220, bins=60, density=True, alpha=0.7, label="$A_{220}$ (fundamental)")
ax.set_xlabel("amplitude (strain)")
ax.set_ylabel("posterior density")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "03_amplitudes.png", dpi=150)
print(f"overtone mean/std = {a221.mean()/a221.std():.2f}, "
      f"median/std = {np.median(a221)/a221.std():.2f}")
