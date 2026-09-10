# Three-stage hierarchical forecasting architecture using MSTL, residual learning (LSTM), and empirical MinT: applied in New Zealand's residential sector

Code and data for *"Hierarchical forecasting of residential construction activity:
a three-stage architecture combining seasonal-trend decomposition, deep residual
learning, and minimum trace reconciliation"*, **Expert Systems With Applications**
331 (2026) 133273.

**Paper:** https://doi.org/10.1016/j.eswa.2026.133273 (open access, CC BY 4.0)

---

## Application

The script was develop to forecast New Zealand's residential construction activity across three hierarchies (
number of consents, gross floor area, and consented capital value ) each split into
detached houses, townhouses and apartments. Total time-series: 12 (9 typology-level and 3
top-level aggregates). The dataset is built from 35 years of monthly building consent data.

The architecture has three stages:

1. **MSTL** decomposes each series and forecasts trend and seasonality (MSTL is chosen after various statistical models are tested)
2. **LSTM** learns the non-linear structure left in the MSTL residuals, applied only
   to the series that were found to have substantial signal left in the residuals (failed a Ljung-Box test at lag 12)
3. **Empirical MinT** reconciliation restores coherence between the typology-level
   forecasts and their aggregates, while increasing accuracy.

The proposed three stage architecture was found to reduce MAE by 23.3–39.7% against the MSTL baseline (across the 12 features) and 6.0–28.5% against
the two-stage MSTL+LSTM hybrid.

## Files

| File | What it does |
|---|---|
| `Script1_Statistical_Modelling.py` | Compares 9 statistical models across all 12 features |
| `Script2_Residual_Analysis.py` | Ljung-Box tests and residual diagnostics |
| `Script3_Bayesian_Optimization_for_hybrid_MSTL_LSTM.py` | Optuna search for the residual learner |
| `Script4_All_models_compared.py` | Full pipeline, ablation study, paper figures |
| `data_9features.xlsx` | Monthly series, 04/1990 – 12/2025 |

Data built from Statistics New Zealand's *Regional new dwellings consented* release
(https://www.stats.govt.nz), accessed 27 November 2025, subject to the Stats NZ
licence terms.

## Running the script

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

Script 4 doesn't need Script 3. The tuned hyperparameters are hardcoded in
it. Only run Script 3 if you're re-tuning.

## Important Notes for proper implementation

**Pin the library versions.** `statsforecast`, `neuralforecast` and
`hierarchicalforecast` change their APIs between releases. The versions in
`requirements.txt` are what these scripts were written against. `statsforecast` 2.x
treats `unique_id` as a column rather than the DataFrame index, and the scripts
assume that, so a 1.x install will fail straight away.

**Script 3 will take hours if attempted to run.** It goes through 1,000 Optuna trials × 3 hierarchy groups × 4 folds, each
one training an LSTM.

**Expect small numerical drift.** The LSTM is seeded and thread counts are pinned,
but exact reproducibility in PyTorch still depends on hardware.

**Clear the caches when you change things.** The scripts write joblib caches
(`sf_cache/` from Scripts 2–4, `statsforecast_cache/` from Script 1). If data or the CV parameters are altered without the caches deleted, you'll get
the old fits back without notification.


## Limitations
This uses historical data only — no interest rates, material costs or population
growth. The residual learner is restricted to sequence-to-sequence LSTMs, so whether
that stage is optimal is untested. Validation is on New Zealand data alone, so the
size of the gains elsewhere is an open question. Section 3.6 of the paper goes
through all of this properly.

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

Gerasimos Christoforatos — gerr.christof@gmail.com

## Acknowledgements

The paper was part of PhD research that was funded by the New Zealand Ministry of Business, Innovation and Employment under the
Āmiomio Aotearoa project (UOWX2004, University of Waikato) and the Building Research
Levy. Built on the Nixtla `statsforecast`, `neuralforecast` and `hierarchicalforecast`
libraries.
