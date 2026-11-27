# Causal Robust Filtering for Zero-Shot Time-Series Foundation Models under Heavy-Tailed Noise

[![Model](https://img.shields.io/badge/Model-TimesFM--2.5--200M-blue)](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Empirical benchmark evaluating the inference stability of **`google/timesfm-2.5-200m-pytorch`** under heavy-tailed Student-$t$ ($\nu = 2$) observational noise, and characterizing the **regime crossover** of a strictly causal, current-inclusive rolling **Median–IQR clipping filter** ($W = 24, k = 1.5$) across temporal resolutions and evaluation settings.

---

## Key Findings

![Regime Crossover](regime_crossover )

* **Sharp Regime Crossover:**
  * **Low-Noise Regime ($\alpha < \hat{\alpha}^*$):** Unconditional Median–IQR clipping degrades zero-shot accuracy ($\Delta\text{MAE} > 0\%$, $p_{\text{Holm}} < 0.05$ for $\alpha \in [0.25, 0.75]$ on `ETTh1` and $\alpha \in [0.25, 0.50]$ on `ETTm1`) by truncating legitimate signal peaks.
  * **High-Noise Regime ($\alpha > \hat{\alpha}^*$):** Once heavy-tailed corruption exceeds the estimated break-even regime, causal pre-filtering yields statistically significant reductions in Mean Absolute Error ($p_{\text{Holm}} \le 0.0034$ for $\alpha \ge 1.25$; $p_{\text{Holm}} < 0.0001$ for $\alpha \ge 1.50$).
* **Benchmark & Resolution Differences:**
  * **ETTh1 (Hourly, $L=448, H=96$, $4$-day horizon, $N=117$):** Estimated linear crossover at $\hat{\alpha}^* \approx \mathbf{1.0416}$, achieving **-16.98%** MAE reduction at $\alpha = 2.00$ ($t = +12.25, p_{\text{Holm}} < 0.0001$).
  * **ETTm1 (15-Minute, $L=448, H=96$, $24$-hour horizon, $N=120$):** Earlier estimated linear crossover at $\hat{\alpha}^* \approx \mathbf{0.8147}$, achieving **-21.55%** MAE reduction at $\alpha = 2.00$ ($t = +13.16, p_{\text{Holm}} < 0.0001$).

---

## Summary Results ($n = 10$ Seeds, $L = 448, H = 96$)

| Noise Scale ($\alpha$) | ETTh1 Raw MAE | ETTh1 Filtered MAE | ETTh1 $\Delta\text{MAE}$ | ETTm1 Raw MAE | ETTm1 Filtered MAE | ETTm1 $\Delta\text{MAE}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | $0.1766 \pm 0.0000$ | $0.1766 \pm 0.0000$ | +0.02% | $0.1279 \pm 0.0000$ | $0.1299 \pm 0.0000$ | +1.54% |
| **0.25** | $0.1941 \pm 0.0024$ | $0.1976 \pm 0.0030$ | +1.83% | $0.1483 \pm 0.0026$ | $0.1515 \pm 0.0040$ | +2.14% |
| **0.50** | $0.2014 \pm 0.0037$ | $0.2116 \pm 0.0038$ | +5.05% | $0.1639 \pm 0.0036$ | $0.1699 \pm 0.0070$ | +3.64% |
| **0.75** | $0.2133 \pm 0.0057$ | $0.2228 \pm 0.0052$ | +4.44% | $0.1803 \pm 0.0050$ | $0.1827 \pm 0.0082$ | +1.35% |
| **1.00** | $0.2284 \pm 0.0077$ | $0.2303 \pm 0.0055$ | +0.84% | $0.1986 \pm 0.0061$ | $0.1909 \pm 0.0090$ | **-3.86%** |
| **1.25** | $0.2458 \pm 0.0097$ | $0.2355 \pm 0.0059$ | **-4.20%** | $0.2187 \pm 0.0071$ | $0.1979 \pm 0.0095$ | **-9.48%** |
| **1.50** | $0.2653 \pm 0.0115$ | $0.2411 \pm 0.0082$ | **-9.13%** | $0.2404 \pm 0.0080$ | $0.2061 \pm 0.0102$ | **-14.28%** |
| **1.75** | $0.2863 \pm 0.0132$ | $0.2481 \pm 0.0104$ | **-13.36%** | $0.2636 \pm 0.0088$ | $0.2154 \pm 0.0113$ | **-18.28%** |
| **2.00** | $0.3087 \pm 0.0147$ | $0.2563 \pm 0.0121$ | **-16.98%** | $0.2879 \pm 0.0095$ | $0.2259 \pm 0.0124$ | **-21.55%** |

---

## Repository Structure

```text
├── notebooks/                  # Evaluation notebooks & statistical audit scripts
├── paper/                      # LaTeX manuscript source and compiled PDF
├── regime_crossover.pdf        # Publication vector figure (Figure 1)
├── regime_crossover.png        # High-resolution plot preview
├── experiment_metadata.json    # Deterministic hyperparameters, dataset splits & seed schedule
└── README.md
