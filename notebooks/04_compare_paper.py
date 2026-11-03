# %% [markdown]
# # Step 4 – Check against Isi et al. 2019
#
# The docs example (Step 3) is *not* meant to reproduce the paper: the docs say
# so explicitly. Here we also run a **paper-like** analysis at the same start
# time ($t_0 = t_{\rm peak}$ = GPS 1126259462.423 at H1), with the settings
# Isi & Farr published with their 2022 reply (4096 s of data, flat amplitude
# prior, nearly face-off inclination). See `common.py` and
# `papers/PAPER_NUMBERS.md` for where each setting comes from.
#
# **How the paper measured significance** (Isi+ 2019, Fig. 2 caption): "the
# mean of the $A_1$ marginalized posterior lies 3.6 standard deviations away
# from zero". So significance = mean($A_1$) / std($A_1$). Isi & Farr 2022 use
# the median instead: median($A_1$) / std($A_1$). We compute both.

# %%
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from common import (configure_jax, make_paper_fit, significance, contour90,
                    FIGS, RESULTS)

configure_jax()

NC = RESULTS / "04_paper_like_t0.nc"
if NC.exists():
    # reuse a finished run (delete the file to sample again)
    paper = az.from_netcdf(NC)
    print(f"loaded {NC.name}")
else:
    t = time.time()
    fit = make_paper_fit(dt0_M=0.0)
    fit.run()
    print(f"paper-like 220+221 at t_ref: {time.time() - t:.0f} s")
    fit.result.to_netcdf(NC)
    paper = fit.result

# %%
docs = az.from_netcdf(RESULTS / "03_docs_220+221.nc")


def summarize(post):
    m = post["m"].values.ravel()
    chi = post["chi"].values.ravel()
    a1 = post["a"].isel(mode=1).values.ravel()
    q = lambda x: np.percentile(x, [16, 50, 84])
    mean_sig, med_sig = significance(a1)
    return {k: float(v) for k, v in dict(
        M_median=q(m)[1], M_lo=q(m)[1] - q(m)[0], M_hi=q(m)[2] - q(m)[1],
        M_mean=m.mean(), M_std=m.std(),
        chi_median=q(chi)[1], chi_lo=q(chi)[1] - q(chi)[0], chi_hi=q(chi)[2] - q(chi)[1],
        chi_mean=chi.mean(), chi_std=chi.std(),
        A1_mean=a1.mean(), A1_median=np.median(a1), A1_std=a1.std(),
        sig_mean_over_std=mean_sig, sig_median_over_std=med_sig,
    ).items()}


rows = {"docs settings (Step 3)": summarize(docs.posterior),
        "paper-like settings": summarize(paper.posterior)}
json.dump(rows, open(RESULTS / "04_summary.json", "w"), indent=1)

print(f"{'':28s} {'M (Msun)':>16s} {'chi':>14s} {'A1 mean/std':>12s} {'A1 med/std':>11s}")
print(f"{'Isi+ 2019 (paper)':28s} {'68 ± 7':>16s} {'0.63 ± 0.16':>14s} {'3.6':>12s} {'-':>11s}")
for k, r in rows.items():
    print(f"{k:28s} {r['M_mean']:8.1f} ± {r['M_std']:4.1f}   "
          f"{r['chi_mean']:5.2f} ± {r['chi_std']:4.2f} "
          f"{r['sig_mean_over_std']:12.2f} {r['sig_median_over_std']:11.2f}")

# %% [markdown]
# The paper quotes "$(68 \pm 7) M_\odot$ and $0.63 \pm 0.16$, with 68%
# credibility" (Isi+ 2019 main text, line 160 of the LaTeX source), so we
# print mean ± standard deviation, which is the closest like-for-like summary.

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for (name, post), c in zip([("docs settings", docs.posterior),
                            ("paper-like settings", paper.posterior)], ["C0", "C2"]):
    contour90(axs[0], post["m"].values.ravel(), post["chi"].values.ravel(), c, name)
    a1 = post["a"].isel(mode=1).values.ravel()
    axs[1].hist(a1, bins=60, density=True, histtype="step", lw=2, color=c,
                label=f"{name}: mean/std = {a1.mean()/a1.std():.1f}")
axs[0].errorbar(68, 0.63, xerr=7, yerr=0.16, fmt="ks", label="Isi+ 2019: 68±7, 0.63±0.16")
axs[0].set_xlabel(r"$M$ ($M_\odot$, detector frame)")
axs[0].set_ylabel(r"$\chi$")
axs[0].set_title("Remnant: 90% regions vs paper (68% bars)")
axs[0].legend(fontsize=8)
axs[1].axvline(0, c="k", lw=1)
axs[1].set_xlabel("overtone amplitude $A_{221}$")
axs[1].set_title("Overtone amplitude at $t_0=t_{\\rm peak}$ (paper: 3.6σ)")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "04_vs_paper.png", dpi=150)
