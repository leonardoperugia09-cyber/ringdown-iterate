# Reference numbers from the papers (with exact locations)

Every number this project compares against comes from this file. Each one is
copied from the authors' own LaTeX source, downloaded from arXiv
(`https://arxiv.org/e-print/<id>`), and the file + line is given so you can check it.

## Isi et al. 2019 — arXiv:1905.00869 (PRL 123, 111102)

Source: `src_1905.00869/qnmot_gw150914.tex` (main text) and `src_1905.00869/supplement.tex`.

| Quantity | Value | Where |
|---|---|---|
| Overtone significance | "$A_1=0$ is disfavored at 3.6σ": the **mean** of the $A_1$ posterior lies 3.6 standard deviations from zero (N=1 model) | Fig. 2 caption, main .tex line 233 |
| Remnant mass, N=1 at peak | $(68 \pm 7)\,M_\odot$, **detector frame**, 68% credibility | main .tex line 160 |
| Remnant spin, N=1 at peak | $0.63 \pm 0.16$, 68% credibility | main .tex line 160 |
| IMR peak (for comparison) | $M_f = 68.5\,M_\odot$, $\chi_f = 0.69$ | Fig. 1 caption, line 208 |
| Start time | $t_0 = t_{\rm peak} = 1126259462.423$ GPS **at Hanford (H1)** | line 100 and 217 |
| N=0 matches IMR at | $\Delta t_0 \approx 3$ ms after peak | line 162, 268 |
| Post-peak network SNR | ~14 (for f > 154.7 Hz) | line 257 |
| δf₁ (overtone frequency deviation) | $-0.05 \pm 0.2$ (68%) | line 276 |
| Sample rate / duration | 2048 Hz, 0.5 s of data from the peak | supplement line 103–104 |
| Noise estimate | autocovariance from 64 s of off-source data, after 20 Hz high-pass (4th-order Butterworth) | supplement lines 48, 98–99 |
| Sky / orientation | ra = 1.95 rad, dec = −1.27 rad, ψ = 0.82 rad, ι = π | supplement line 113 |
| L1 time shift | 7 ms arrival-time delay | supplement line 115 |
| Priors | $A_n \in [0, 2.5\times10^{-19}]$, $\phi_n \in [0,2\pi]$, $M_f \in [50,100]\,M_\odot$, $\chi_f \in [0,1]$, uniform | supplement lines 119–122 |
| Sampler | kombine | supplement line 124 |

## Cotesta et al. 2022 — arXiv:2201.00822 (PRL 129, 111102)

Source: `src_2201.00822/paper_prl.tex`.

| Quantity | Value | Where |
|---|---|---|
| Their peak-time estimate (H1) | median $\bar t_{\rm peak} = 1126259462.42323$ s, σ = 0.00059 s (SEOBNRv4) | line 220 |
| Alternative (IMRPhenomPv2) | median 1126259462.42371 s, σ = 0.00063 s | Supplement, line 332 |
| Scan range | $t_{\rm start}$ varied within ±2σ of the peak posterior | line 221 |
| Time unit | Δt quoted in units of $M$; e.g. "0.24 ms ≈ 0.72 M" | line 236 |
| Isi 2019's start time in their units | $\Delta t/M = -0.72$ | lines 231, 236 |
| Bayes factor at Isi's time | $\log_{10}\mathcal B^{221}_{220} = -0.60$ | line 236 |
| Strong evidence only when | $\Delta t/M \le -4.34$ (well before the peak) | line 236 |
| Overtone damping time | $\tau_{221} \approx 1.3$ ms $\approx 4M$ | line 236 |
| Priors | $M_f\in[20,200]$, $a_f\in[0,0.99]$, $A\in[0,5\times10^{-20}]$ | line 189 |
| Settings | pyRing + cpnest, 16384 Hz, 0.1 s of data, ACF from 64 s starting at GPS 1126257417 | lines 204–213 (ACF: line 209) |
| Third reanalysis they cite (Finch & Moore) | 1.8σ | line 278 |

## Isi & Farr 2022 — arXiv:2202.02941 (reply)

Source: `src_2202.02941/rd_redux.tex`, plus their released notebooks
(Zenodo 10.5281/zenodo.5965773) in `isi_farr_2022_release/`.

| Quantity | Value | Where |
|---|---|---|
| Time unit | $t_M = GM/c^3$ with $M = 69\,M_\odot$, i.e. $t_M = 0.34$ ms | Fig. 1 caption, line 155 |
| Reference time | $t_{\rm ref} = 1126259462.423$ s at H1 | Fig. 1 caption |
| Significance measure | **median / standard deviation** of the $A_1$ posterior | Fig. 1 caption, text |
| Start-time grid | $t_0 - t_{\rm ref} \in \{-1.0, -0.8, \dots, 4.0\}\,t_M$ plus $0$ and $0.68\,t_M$ (Cotesta's preferred time) | `paper_plots.ipynb`, cell 5 |
| Data | 4096 s files at 16 kHz, downsampled to 2048 Hz, 0.2 s analysed | footnote; `GW150914_sample_analysis.ipynb` cells 8, 16 |
| Model | `mchi_aligned`, two tones, $\cos\iota \le -0.99$, flat prior on $A$ | footnote; notebook cell 30 |
| Priors | `A_scale=5e-21, M_min=35, M_max=140, cosi_max=-0.99, flat_A=1` | notebook cell 30 |
| Injections | SXS:BBH:0305 in real noise give ~3.5σ on average, ± ~1σ scatter | line 268 |
| Kerr-model breakdown | sharp transition around $t_0 \approx -1.5\,t_M$ | line 212 |

## ringdown docs GW150914 example

Source: https://ringdown.readthedocs.io/en/latest/examples/GW150914.html

| Quantity | Value |
|---|---|
| Target | `set_target(1126259462.4083147, ra=1.95, dec=-1.27, psi=0.82, duration=0.2)` (geocentre time; equals 1126259462.423 at H1) |
| Conditioning | `condition_data(ds=8, f_min=20)` → 2048 Hz |
| Priors | `update_model(a_scale_max=5e-20, m_min=40.0, m_max=140.0, cosi=-1)` |
| Data | 32 s, 16 kHz GWOSC files; MD5 H1 `b345645c0488287301824fe617f58dc2`, L1 `5eb246ff21364d38471c95452f012853` |
| Docs results (220+221) | m = 72.1 ± 8.2, χ = 0.68 ± 0.17 |
| Docs caveat | Not meant to reproduce Isi et al. 2019 exactly (different polarization model, priors, conditioning) |
