# Conditional Robustness of Zero-Shot Time-Series Foundation Models: Regime-Dependent Preprocessing Under Heavy-Tailed Noise

> **Status:** Research-in-Progress — Experimental sweeps and statistical verification on `ETTh1` and `ETTm1` are complete; manuscript compilation and controlled downsampling ablations are ongoing.

An empirical investigation into when causal input preprocessing improves versus degrades zero-shot forecasting in `TimesFM-2.5-200M` (`google/timesfm-2.5-200m-pytorch`) under heavy-tailed sensor corruption.

---

## 1. Research Question

Pretrained time-series foundation models apply per-context input normalization before transformer encoding. Extreme outliers can distort context-scale statistics, potentially affecting forecasting accuracy under impulsive, heavy-tailed noise.

Rather than assuming that denoising is universally beneficial, this project investigates:

> **When does causal input preprocessing improve versus degrade zero-shot forecasting accuracy under heavy-tailed noise, and how does the break-even threshold vary across datasets with different temporal resolutions?**

---

## 2. Central Empirical Findings

### 2.1 Regime-Dependent Preprocessing Crossover
Causal Rolling Median-IQR clipping ($W=24, k=1.5$) is **conditionally beneficial rather than universally robust**. The same filter significantly *degrades* zero-shot forecasting under moderate heavy-tailed noise ($\alpha \le 1.00$ on `ETTh1`), potentially because clipping removes genuine signal extrema. It reaches statistical parity in an intermediate transition zone and produces statistically significant error reductions under severe corruption ($\alpha = 2.00$ on `ETTh1` and $\alpha \ge 1.25$ on `ETTm1`).

### 2.2 Preliminary Spectral and Polynomial Filter Screening
In preliminary screening experiments conducted exclusively on `ETTh1` (`OT`, $\alpha \in [0.1, 1.0]$), frequency-domain FFT low-pass filtering and polynomial Savitzky–Golay smoothing consistently degraded zero-shot accuracy compared with unfiltered inputs. These spectral baselines were evaluated only on `ETTh1` and were not replicated across `ETTm1`. Therefore, subsequent cross-dataset experiments focused on causal time-domain Median-IQR clipping.

### 2.3 Sampling-Resolution-Associated Crossover Shift
The estimated zero-benefit threshold differs between the two evaluated datasets:
- **`ETTh1` (hourly):** $\alpha^* \approx 1.43$
- **`ETTm1` (15-minute):** $\alpha^* \approx 0.93$

This suggests that the effectiveness of local robust clipping may depend on temporal resolution and local autocorrelation characteristics. However, these results do not establish that sampling frequency causes the difference. Controlled experiments, such as downsampling `ETTm1` to hourly intervals on the same underlying physical series, are required to isolate the contribution of sampling frequency from dataset-specific variance.

---

## 3. Statistical Evidence

### 3.1 Experimental Unit and Metric Definition
To avoid pseudo-replication across evaluation windows, forecast errors are first averaged across all $N=30$ (`ETTh1`) or $N=120$ (`ETTm1`) test windows within each random seed. All hypothesis tests operate strictly on the **$n=10$ paired seed-level observations** ($\text{df}=9$) per noise level $\alpha$.

Relative error change is defined as:

$$\Delta\text{MAE}(\%) = \frac{\text{MAE}_{\text{clipped}} - \text{MAE}_{\text{raw}}}{\text{MAE}_{\text{raw}}} \times 100$$

**Interpretation:**
- **Positive ($\Delta\text{MAE} > 0$):** Clipping degrades forecasting accuracy relative to unfiltered noisy input.
- **Negative ($\Delta\text{MAE} < 0$):** Clipping improves forecasting accuracy.
- **Zero ($\Delta\text{MAE} = 0$):** Break-even parity, used to estimate the crossover threshold $\alpha^*$ through piecewise linear interpolation between adjacent evaluated grid points.

### 3.2 Cross-Dataset Heavy-Tailed Sweep ($\nu=2$)

**Table 1.** Cross-resolution zero-shot MAE ($\text{Mean} \pm \text{SD}$ across 10 paired seeds, $H=96$) under Student-$t$ ($\nu=2$) corruption. The $p_{\text{Holm}}$ values report Holm–Bonferroni-adjusted two-sided paired $t$-test $p$-values across the 7 noise levels.

| $\alpha$ | `ETTh1` Unfiltered ($N=30$) | `ETTh1` Median-IQR ($N=30$) | `ETTh1` $\Delta\text{MAE}$ | `ETTh1` $p_{\text{Holm}}$ | `ETTm1` Unfiltered ($N=120$) | `ETTm1` Median-IQR ($N=120$) | `ETTm1` $\Delta\text{MAE}$ | `ETTm1` $p_{\text{Holm}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | $1.818 \pm 0.081$ | $1.936 \pm 0.103$ | $+6.45\%$ | **0.0155** | $1.465 \pm 0.047$ | $1.522 \pm 0.068$ | $+3.90\%$ | 0.0604 |
| **0.75** | $1.899 \pm 0.099$ | $2.053 \pm 0.104$ | $+8.09\%$ | **0.0017** | $1.578 \pm 0.061$ | $1.621 \pm 0.084$ | $+2.77\%$ | 0.1599 |
| **1.00** | $2.005 \pm 0.114$ | $2.154 \pm 0.111$ | $+7.42\%$ | **0.0344** | $1.708 \pm 0.075$ | $1.689 \pm 0.099$ | $-1.12\%$ | 0.4640 |
| **1.25** | $2.133 \pm 0.130$ | $2.206 \pm 0.135$ | $+3.45\%$ | 0.4423 | $1.856 \pm 0.087$ | $1.756 \pm 0.107$ | **$-5.39\%$** | **0.0211** |
| **1.50** | $2.278 \pm 0.145$ | $2.247 \pm 0.147$ | $-1.36\%$ | 0.6338 | $2.019 \pm 0.100$ | $1.831 \pm 0.115$ | **$-9.31\%$** | **0.0006** |
| **1.75** | $2.436 \pm 0.161$ | $2.295 \pm 0.160$ | $-5.80\%$ | 0.2166 | $2.194 \pm 0.111$ | $1.916 \pm 0.121$ | **$-12.66\%$** | **0.0001** |
| **2.00** | $2.604 \pm 0.177$ | $2.356 \pm 0.177$ | **$-9.53\%$** | **0.0429** | $2.379 \pm 0.122$ | $2.009 \pm 0.129$ | **$-15.55\%$** | **<0.0001** |

### 3.3 Interpolated Crossover Thresholds
The estimated break-even thresholds, obtained through linear interpolation, are:
- **`ETTh1`:** $\alpha^* = 1.4294$, interpolated between $\alpha = 1.25$ and $\alpha = 1.50$.
- **`ETTm1`:** $\alpha^* = 0.9283$, interpolated between $\alpha = 0.75$ and $\alpha = 1.00$.

These are interpolated estimates, not directly tested noise levels.

---

## 4. Repository Structure

```text
ts-robustness-benchmark/
├── LICENSE
├── README.md
├── notebooks/
│   └── timesfm_robustness_experiments.ipynb
└── paper/
    ├── main.tex
    ├── table_cross_dataset.tex
    └── fig_cross_dataset_crossover.pdf
