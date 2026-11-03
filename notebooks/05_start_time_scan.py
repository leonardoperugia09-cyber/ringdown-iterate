# %% [markdown]
# # Step 5 – The start-time scan
#
# The same 220+221 fit (paper-like settings) repeated at many start times
# $t_0$ around the peak. Results come from `05a_run_scan.py`.
#
# **Units.** Times are measured from $t_{\rm ref}$ = GPS 1126259462.423 at
# Hanford (the start time of Isi+ 2019), in units of $t_M = GM/c^3$, the time
# light takes to cross (half) the black hole's size. Isi & Farr 2022 use
# $M = 69\,M_\odot$, giving $t_M = 0.34$ ms (their Fig. 1 caption). So
# "+1 $t_M$" means 0.34 ms later. Cotesta+ 2022 also quote times "in units of
# $M$" (e.g. "0.24 ms ≈ 0.72 M"), but relative to *their* peak estimate,
# which is 0.68 $t_M$ after $t_{\rm ref}$.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from common import FIGS, RESULTS, t_M_seconds

df = pd.read_csv(RESULTS / "05_scan_summary.csv").sort_values("dt0_M")
tM_ms = t_M_seconds() * 1e3
print(f"t_M = {tM_ms:.3f} ms")
df[["dt0_M", "dt0_ms", "A1_median", "A1_std", "sig_mean_over_std", "sig_median_over_std"]]

# %%
# Cotesta+ 2022 peak-time estimate (paper_prl.tex line 220): median 1126259462.42323 s,
# sigma 0.59 ms -> in t_M from t_ref:
cot_med = (1126259462.42323 - 1126259462.423) * 1e3 / tM_ms
cot_sig = 0.59 / tM_ms

fig, ax = plt.subplots(figsize=(10, 5))
ax.axvspan(cot_med - 2 * cot_sig, cot_med + 2 * cot_sig, color="gold", alpha=0.15,
           label="Cotesta+ 2022 peak estimate ±2σ")
ax.axvspan(cot_med - cot_sig, cot_med + cot_sig, color="gold", alpha=0.3,
           label="Cotesta+ 2022 peak estimate ±1σ")
ax.plot(df.dt0_M, df.sig_median_over_std, "o-", c="C0", label="my result: median($A_1$)/std($A_1$)")
ax.plot(df.dt0_M, df.sig_mean_over_std, "s--", c="C1", ms=4, alpha=0.7,
        label="my result: mean($A_1$)/std($A_1$)")
ax.axvline(0, c="k", lw=1)
ax.text(0.05, ax.get_ylim()[1] * 0.97, "  $t_{\\rm ref}$ (Isi+ 2019)", va="top")
ax.axvline(cot_med, c="goldenrod", lw=1, ls="--")
ax.plot([0], [3.6], "k*", ms=14, label="Isi+ 2019: 3.6σ at $t_{\\rm ref}$")
ax.axhline(0, c="gray", lw=0.5)
ax.set_xlabel(r"start time $t_0 - t_{\rm ref}$  [$t_M$, M = 69 $M_\odot$]")
ax.set_ylabel("overtone significance (σ)")
sec = ax.secondary_xaxis("top", functions=(lambda x: x * tM_ms, lambda x: x / tM_ms))
sec.set_xlabel("start time $t_0 - t_{\\rm ref}$  [ms]")
ax.set_title("GW150914: overtone significance vs ringdown start time", pad=40)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIGS / "05_significance_vs_start_time.png", dpi=150)

# %% [markdown]
# ## The full amplitude posteriors
# The significance number hides the shape of the posterior. Here are the
# $A_1$ posteriors themselves (median and 68%/90% ranges), plus how fast the
# overtone *should* decay if it is real: $A_1(t_0) = A_1(0)\,e^{-t_0/\tau_{221}}$,
# using the measured $\tau_{221}$ from the $t_{\rm ref}$ run.

# %%
ref = az.from_netcdf(RESULTS / "scan" / "scan_dt0M_+0.00.nc").posterior
a1_ref = ref["a"].isel(mode=1).values.ravel()
g1_ref = ref["g"].isel(mode=1).values.ravel()  # damping rate gamma = 1/tau (Hz)
tau221_ms = np.median(1 / g1_ref) * 1e3
print(f"tau_221 at t_ref (median) = {tau221_ms:.2f} ms = {tau221_ms/tM_ms:.1f} t_M")

fig, ax = plt.subplots(figsize=(10, 5))
for _, r in df.iterrows():
    p = az.from_netcdf(RESULTS / "scan" / f"scan_dt0M_{r.dt0_M:+.2f}.nc").posterior
    a1 = p["a"].isel(mode=1).values.ravel()
    lo5, lo16, med, hi84, hi95 = np.percentile(a1, [5, 16, 50, 84, 95])
    ax.plot([r.dt0_M] * 2, [lo5, hi95], c="C0", lw=1)
    ax.plot([r.dt0_M] * 2, [lo16, hi84], c="C0", lw=4)
    ax.plot(r.dt0_M, med, "o", c="w", mec="C0")
x = np.linspace(0, df.dt0_M.max(), 100)
decay = np.exp(-np.outer(x * tM_ms * 1e-3, g1_ref[:2000]))
env = np.percentile(a1_ref[:2000] * decay, [16, 84], axis=1)
ax.fill_between(x, env[0], env[1], color="gray", alpha=0.3,
                label="expected decay of $A_1$ from the $t_{\\rm ref}$ fit (68%)")
ax.axvline(0, c="k", lw=1)
ax.axhline(0, c="gray", lw=0.5)
ax.set_xlabel(r"$t_0 - t_{\rm ref}$  [$t_M$]")
ax.set_ylabel("overtone amplitude $A_1$ (strain)")
ax.set_title("$A_1$ posterior vs start time (dots: median; bars: 68% and 90%)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "05_A1_vs_start_time.png", dpi=150)

# %% [markdown]
# Export the scan for the project page's interactive start-time slider.

# %%
import json
cols = ["dt0_M", "dt0_ms", "sig_mean_over_std", "sig_median_over_std", "A1_q05", "A1_q16",
        "A1_median", "A1_q84", "A1_q95", "M_median", "chi_median"]
out = dict(t_M_ms=float(tM_ms), cotesta_median_tM=float(cot_med), cotesta_sigma_tM=float(cot_sig),
           tau221_ms=float(tau221_ms),
           A1_unit=1e-21,  # amplitudes below are in units of 1e-21 strain
           rows=[{k: round(float(v) / (1e-21 if k.startswith("A1") else 1), 6) for k, v in r.items()}
                 for r in df[cols].to_dict(orient="records")])
(RESULTS.parent / "docs").mkdir(exist_ok=True)
# a .js file (not .json) so the page also works when opened straight from disk
with open(RESULTS.parent / "docs" / "scan.js", "w") as f:
    f.write("window.SCAN = " + json.dumps(out, indent=1) + ";\n")
print(f"wrote docs/scan.js with {len(df)} start times")
