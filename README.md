# Zero-Shot Time-Series Foundation Model Robustness Under Heavy-Tailed Noise

An empirical robustness benchmark evaluating Google Research's **TimesFM-2.5-200M** (`google/timesfm-2.5-200m-pytorch`) on Electricity Transformer Temperature datasets (**ETTh1**, **ETTm1**) under synthetic Gaussian and heavy-tailed Student-$t$ ($\nu=2$) noise corruption.

## Key Findings

* **Regime Crossover Behavior:** Causal preprocessing introduces slight smoothing distortion in clean or low-noise regimes ($\alpha < \alpha^*$), but yields substantial error reductions once heavy-tailed noise intensity crosses the break-even threshold ($\alpha > \alpha^*$).
* **Verified Crossover Thresholds ($\alpha^*$):**
  * **ETTh1 (Hourly, Normalized 12/4/4 Split, $L=448, H=96$):** $\alpha^* \approx 1.0391$
  * **ETTm1 (15-Min, Normalized 12/4/4 Split, $L=448, H=96$):** $\alpha^* \approx 0.8102$
  * **Unnormalized Scale Sweep ($L=512, H=96$, Holm-corrected):** $\alpha^* \approx 1.43$ on ETTh1 ($-9.53\%$ MAE at $\alpha=2.0$) and $\alpha^* \approx 0.93$ on ETTm1 ($-15.55\%$ MAE at $\alpha=2.0$).
* **Strict Causality:** Unlike Savitzky-Golay smoothing or global FFT low-pass filtering, causal **Rolling Median-IQR clipping** ($W=24, k=1.5$) operates strictly on past observations without lookahead leakage.

## Repository Structure

```text
├── TS-Robust.ipynb              # End-to-end Colab benchmark & statistical evaluation
├── experiment_metadata.json     # Environment versions, hyperparameters, and verified thresholds
├── table_cross_dataset.tex      # Camera-ready LaTeX summary table
├── generate_crossover_plot.py   # Reproducible script to render regime_crossover.pdf
└── regime_crossover.pdf         # Multi-panel regime crossover visualization
