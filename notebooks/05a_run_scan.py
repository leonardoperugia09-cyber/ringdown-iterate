"""Step 5 runner: the 220+221 paper-like fit at many start times.

Long job, so it is a plain script (run it in the background) and it is
resumable: start times whose result file already exists are skipped.

    python notebooks/05a_run_scan.py            # all start times, then the summary
    python notebooks/05a_run_scan.py 1/2        # worker 1 of 2 (run 2/2 in parallel)
    python notebooks/05a_run_scan.py next       # fit one missing start time, then exit

Start times are run in order of closeness to t_ref, so partial results already
cover the most important region. One fit needs ~4 GB of RAM, and memory grows
if several fits run in the same process, so on a small machine use `next` in a
loop (run.sh does this): every fit gets a fresh process.

Start-time grid (units of t_M = G M/c^3, M = 69 Msun, t_M = 0.34 ms; relative
to t_ref = GPS 1126259462.423 at H1):
  * Isi & Farr 2022 grid: -1.0 ... 4.0 in steps of 0.2, plus 0 and 0.68
    (paper_plots.ipynb, cell 5).
  * extended down to -2.8 so it covers Cotesta+ 2022's ±2σ peak-time range:
    their median peak is 0.23 ms (= 0.68 t_M) after t_ref with σ = 0.59 ms
    (= 1.74 t_M), so ±2σ spans -2.8 ... +4.2 t_M (paper_prl.tex line 220).
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (configure_jax, make_paper_fit, significance, fetch,
                    FILES_4096S, RESULTS, t_M_seconds)

configure_jax()

GRID = sorted(set([round(x, 2) for x in np.arange(-2.8, 4.2 + 1e-9, 0.2)] + [0.0, 0.68]))
OUT = RESULTS / "scan"
OUT.mkdir(exist_ok=True)
SUMMARY = RESULTS / "05_scan_summary.csv"

paths = fetch(FILES_4096S)
tM = t_M_seconds()

if len(sys.argv) > 1:
    # worker mode: fit every n-th start time (closest to t_ref first), no summary;
    # "next" fits only the first missing one (exit code 3 when none are left)
    order = sorted(GRID, key=lambda x: (abs(x), x))
    if sys.argv[1] == "next":
        todo = [x for x in order if not (OUT / f"scan_dt0M_{x:+.2f}.nc").exists()][:1]
        if not todo:
            sys.exit(3)
        k, n = 1, 1
    else:
        k, n = map(int, sys.argv[1].split("/"))
        todo = order[k - 1::n]
    for dt0 in todo:
        nc = OUT / f"scan_dt0M_{dt0:+.2f}.nc"
        if nc.exists():
            continue
        t = time.time()
        fit = make_paper_fit(dt0_M=dt0, data_paths=paths)
        fit.run()
        fit.result.to_netcdf(nc)
        print(f"[{k}/{n}] dt0 = {dt0:+.2f} t_M done in {time.time() - t:.0f} s", flush=True)
    sys.exit(0)

rows = []
for dt0 in GRID:
    nc = OUT / f"scan_dt0M_{dt0:+.2f}.nc"
    if nc.exists():
        import arviz as az
        post = az.from_netcdf(nc).posterior
        runtime = float("nan")
    else:
        t = time.time()
        fit = make_paper_fit(dt0_M=dt0, data_paths=paths)
        fit.run()
        runtime = time.time() - t
        fit.result.to_netcdf(nc)
        post = fit.result.posterior
    a1 = post["a"].isel(mode=1).values.ravel()
    mean_sig, med_sig = significance(a1)
    row = dict(dt0_M=dt0, dt0_ms=dt0 * tM * 1e3,
               A1_mean=a1.mean(), A1_median=np.median(a1), A1_std=a1.std(),
               A1_q05=np.percentile(a1, 5), A1_q16=np.percentile(a1, 16),
               A1_q84=np.percentile(a1, 84), A1_q95=np.percentile(a1, 95),
               sig_mean_over_std=mean_sig, sig_median_over_std=med_sig,
               M_median=float(np.median(post["m"])), chi_median=float(np.median(post["chi"])),
               runtime_s=runtime)
    rows.append(row)
    print(f"dt0 = {dt0:+.2f} t_M ({row['dt0_ms']:+.3f} ms): "
          f"median/std = {med_sig:.2f}, mean/std = {mean_sig:.2f}, "
          f"took {runtime:.0f} s", flush=True)
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print("done ->", SUMMARY)
