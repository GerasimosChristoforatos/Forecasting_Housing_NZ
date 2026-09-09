# Hierarchical Forecasting of Residential Construction Activity

Code and data for *"Hierarchical forecasting of residential construction activity:
a three-stage architecture combining seasonal-trend decomposition, deep residual
learning, and minimum trace reconciliation"*, **Expert Systems With Applications**
331 (2026) 133273.

**Paper:** https://doi.org/10.1016/j.eswa.2026.133273 (open access, CC BY 4.0)

These are the supplementary files as published, unmodified.

---

## What this does

Forecasts New Zealand residential construction activity across three hierarchies —
number of consents, gross floor area, and consented capital value — each
disaggregated into detached houses, townhouses and apartments. Twelve series in
total: 9 bottom-level and 3 top-level aggregates.

Three stages:

1. **MSTL** decomposes each series and forecasts trend and seasonality.
2. **LSTM** learns the non-linear structure left in the MSTL residuals, applied
   only to the series that failed a Ljung-Box test at lag 12.
3. **Empirical MinT** reconciliation enforces coherence between bottom and top levels.

MAE reductions of 23.3–39.7% against the MSTL baseline, and 6.0–28.5% against the
two-stage MSTL+LSTM hybrid.

## Files

| File | Purpose |
|---|---|
| `Script1_Statistical_Modelling.py` | Compares 9 statistical models across all 12 features |
| `Script2_Residual_Analysis.py` | Ljung-Box tests and residual diagnostics |
| `Script3_Bayesian_Optimization_for_hybrid_MSTL_LSTM.py` | Optuna search for the residual learner |
| `Script4_All_models_compared.py` | Full pipeline, ablation study, paper figures |
| `data_9features.xlsx` | Monthly series, 04/1990 – 12/2025 |

Data constructed from Statistics New Zealand's *Regional new dwellings consented*
release (https://www.stats.govt.nz), accessed 27 November 2025, subject to the
Stats NZ licence terms.

## Running

Python 3.12.7.

```bash
pip install -r requirements.txt
```

Run from the repository root so the scripts find `data_9features.xlsx`:

```bash
python Script1_Statistical_Modelling.py
python Script2_Residual_Analysis.py
python Script3_Bayesian_Optimization_for_hybrid_MSTL_LSTM.py
python Script4_All_models_compared.py
```

Script 4 does not depend on Script 3 — the optimal hyperparameters are already
hardcoded there. Only run Script 3 if you are re-tuning.

## Before you run

**Library versions matter.** `statsforecast`, `neuralforecast` and
`hierarchicalforecast` change APIs between releases. The pinned versions in
`requirements.txt` are the ones these scripts were written against. In particular,
`statsforecast` 2.x treats `unique_id` as a column rather than the DataFrame index,
so a 1.x install will fail immediately.

**Script 1 has a known bug.** The heterogeneous-baseline block at the end calls
`load_and_filter_hierarchy` expecting three return values, but the function returns
two, raising `ValueError: not enough values to unpack`. To run it, change the return
statement to `return hier_df.reset_index(), S_df, tags` and indent the block into
the `if __name__ == "__main__"` guard. This affects only the supplementary
heterogeneous comparison — the main results are unaffected, and Script 4 is correct
as published.

**Script 3 is slow.** 1,000 trials × 3 hierarchy groups × 4 folds, each training an
LSTM. Many hours on CPU, and not resumable if interrupted.

**Results may drift slightly.** The LSTM is seeded, but full reproducibility in
PyTorch also depends on hardware. Published results were produced on CPU, Windows.

**Caching.** The scripts write joblib caches (`sf_cache/`, `statsforecast_cache/`).
Delete these if you change the data or the cross-validation settings, or stale fits
will be returned.

**Fonts.** Scripts 2 and 4 set the font to Cambria. On Linux and macOS matplotlib
falls back to its default and warns. Cosmetic only.

## Limitations

The framework uses endogenous historical data only — no interest rates, material
costs or population growth. The residual learner is restricted to
sequence-to-sequence LSTMs. Validation is on New Zealand data only. Section 3.6 of
the paper covers these in full.

## Citing

```bibtex
@article{Christoforatos2026,
  title   = {Hierarchical forecasting of residential construction activity: a
             three-stage architecture combining seasonal-trend decomposition,
             deep residual learning, and minimum trace reconciliation},
  author  = {Christoforatos, Gerasimos and Pickering, Kim},
  journal = {Expert Systems With Applications},
  volume  = {331},
  pages   = {133273},
  year    = {2026},
  doi     = {10.1016/j.eswa.2026.133273}
}
```

## Licence

Code: MIT. Paper: CC BY 4.0. Data: subject to the Statistics New Zealand licence terms.

## Contact

Gerasimos Christoforatos — gc243@students.waikato.ac.nz
School of Engineering, University of Waikato, Hamilton, New Zealand

## Acknowledgements

Funded by the New Zealand Ministry of Business, Innovation and Employment under the
Āmiomio Aotearoa project (UOWX2004, University of Waikato) and the Building Research
Levy. Built on the Nixtla `statsforecast`, `neuralforecast` and
`hierarchicalforecast` libraries.
