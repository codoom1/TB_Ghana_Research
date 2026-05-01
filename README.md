# Ghana Tuberculosis Trend and Forecasting

This project uses IHME Global Burden of Disease 2023 data for tuberculosis in
Ghana from 1990 to 2023.

The workflow prepares the raw GBD extract, performs exploratory trend and
stationarity analysis, evaluates eight competing forecasting models, and
forecasts the primary TB incidence, mortality, and mortality-to-incidence ratio
endpoints to 2030.

## Repository Structure

```text
.
├── data/
│   ├── TB_df/                         # Raw IHME GBD export and citation
│   └── processed/                     # Clean long/wide modeling datasets
├── docs/
│   └── methods_model_documentation.md # Methods, equations, rationale, citations
├── outputs/
│   ├── figures/
│   │   ├── trend/                     # Initial trend-analysis figures
│   │   ├── eda/                       # EDA, stratified, ACF/PACF figures
│   │   ├── evaluation/                # Holdout prediction and error figures
│   │   ├── forecasts/                 # Point forecast comparison figures
│   │   └── uncertainty/               # Forecast uncertainty figures
│   ├── forecasts/                     # Final forecast CSVs
│   ├── model_outputs/                 # Test-period model predictions
│   └── tables/                        # Summary, diagnostics, metrics, forecasts
├── scripts/
│   ├── 00_eda_stationarity.py         # EDA plots and ADF/KPSS stationarity tests
│   ├── 01_prepare_and_explore.py      # Data preparation and initial trend outputs
│   ├── 02_forecast_classical.py       # Classical-only reference workflow
│   ├── 03_train_evaluate_all_models.py# Full 8-model training/evaluation workflow
│   ├── 04_forecast_best_models.py     # Final best-model and all-model forecasts
│   ├── 05_forecast_uncertainty.py     # Simulation-based forecast uncertainty
│   └── model_*.py                     # Individual model implementations
└── README.md
```

## Environment Setup

Create and activate a conda environment:

```bash
conda env create -f environment.yml
conda activate tb_forecast
```

Alternatively, create the environment manually:

```bash
conda create -n tb_forecast python=3.11 -y
conda activate tb_forecast
conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn statsmodels tqdm -y
pip install torch torchvision
```

For pip-only setup:

```bash
pip install -r requirements.txt
```

For other platforms, install PyTorch using the command recommended for your
machine at https://pytorch.org/get-started/locally/.

Check the installation:

```bash
python - <<'PY'
import pandas
import numpy
import matplotlib
import seaborn
import sklearn
import statsmodels
import torch
import tqdm

print("Libraries loaded successfully")
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
print("CUDA available:", torch.cuda.is_available())
PY
```

## Data

Raw data:

- `data/TB_df/IHME-GBD_2023_DATA-42460a02-1.csv`
- `data/TB_df/citation.txt`

Prepared modeling data:

- `data/processed/ghana_tb_modeling_long.csv`
- `data/processed/ghana_tb_modeling_wide.csv`

The prepared modeling table contains these seven annual series:

- `incidence_age_standardized_rate`
- `mortality_to_incidence_age_standardized_ratio`
- `incidence_all_age_rate`
- `incidence_all_age_number`
- `mortality_age_standardized_rate`
- `mortality_all_age_rate`
- `mortality_all_age_number`

The main analysis now focuses on three age-standardized endpoints:

- `incidence_age_standardized_rate`
- `mortality_age_standardized_rate`
- `mortality_to_incidence_age_standardized_ratio`

The all-age rates and counts remain available as secondary options.

## Quick Start

Activate the project environment first:

```bash
conda activate tb_forecast
```

Then run the full primary analysis:

```bash
# 1. Prepare clean long and wide modeling datasets
python scripts/01_prepare_and_explore.py

# 2. Run EDA, stratified plots, and stationarity tests
python scripts/00_eda_stationarity.py

# 3. Train/evaluate all 8 models on the primary age-standardized endpoints
python scripts/03_train_evaluate_all_models.py --epochs 50

# 4. Refit selected best models and forecast 2024-2030
python scripts/04_forecast_best_models.py --epochs 50

# 5. Propagate GBD uncertainty intervals through the selected best models
python scripts/05_forecast_uncertainty.py --n-sim 100 --epochs 50
```

The default pipeline focuses on the three primary age-standardized endpoints:

```text
incidence_age_standardized_rate
mortality_age_standardized_rate
mortality_to_incidence_age_standardized_ratio
```

To include all seven prepared series, add `--all-series` to steps 3 and 4:

```bash
python scripts/03_train_evaluate_all_models.py --epochs 50 --all-series
python scripts/04_forecast_best_models.py --epochs 50 --all-series
```

To run only a specific endpoint:

```bash
python scripts/03_train_evaluate_all_models.py \
  --epochs 50 \
  --series incidence_age_standardized_rate
```

## Pipeline Steps

Run data preparation and trend exploration:

```bash
python scripts/01_prepare_and_explore.py
```

Run dedicated EDA, stratified plots, and stationarity tests:

```bash
python scripts/00_eda_stationarity.py
```

Run the original classical forecast comparison:

```bash
python scripts/02_forecast_classical.py
```

The classical script evaluates Naive, Drift, ARIMA, and ETS models on a
2018-2023 test set, then refits the selected model on 1990-2023 and forecasts
2024-2030.

Run the full model competition with progress:

```bash
python scripts/03_train_evaluate_all_models.py --epochs 50
```

By default, this evaluates the three primary age-standardized endpoints. To run
all seven series:

```bash
python scripts/03_train_evaluate_all_models.py --epochs 50 --all-series
```

Run final 2024-2030 forecasts from the selected best model for each primary
series:

```bash
python scripts/04_forecast_best_models.py --epochs 50
```

Use `--all-series` with the training and forecast scripts to include secondary
all-age rates and counts.

Run simulation-based forecast uncertainty intervals:

```bash
python scripts/05_forecast_uncertainty.py --n-sim 100 --epochs 50
```

This samples historical trajectories from each GBD `lower`, `val`, and `upper`
interval, refits the selected best model on each simulated trajectory, and
summarizes the 2.5th and 97.5th percentiles of the resulting 2024-2030
forecasts.

## Models

The main model competition evaluates eight models:

```text
Naive
Drift
ARIMA
ETS
LSTM
Causal_TCN
ARIMA_LSTM
Multistep_LSTM
```

Model scripts:

- `scripts/model_arima.py`
- `scripts/model_lstm.py`
- `scripts/model_causal_tcn.py`
- `scripts/model_arima_lstm.py`
- `scripts/model_multistep_lstm.py`
- `scripts/model_ets.py`
- `scripts/model_baselines.py`
- `scripts/model_common.py`

Detailed model definitions, equations, rationale, limitations, and citations
are documented in:

- `docs/methods_model_documentation.md`

## Useful Output Checks

Preview the selected best model per primary endpoint:

```bash
python - <<'PY'
import pandas as pd

best = pd.read_csv("outputs/tables/best_models_by_series.csv")
print(best[["series", "model", "rmse", "mape"]].round(4).to_string(index=False))
PY
```

Preview final forecast endpoints:

```bash
python - <<'PY'
import pandas as pd

fc = pd.read_csv("outputs/forecasts/best_model_forecasts_2024_2030.csv")
endpoints = fc[fc["year"].isin([2024, 2030])]
print(endpoints.pivot(index="series", columns="year", values="forecast").round(3))
PY
```

Preview all model forecasts:

```bash
python - <<'PY'
import pandas as pd

fc = pd.read_csv("outputs/forecasts/all_model_forecasts_2024_2030.csv")
print(fc.head(12).round(3).to_string(index=False))
PY
```

## Outputs

Trend tables:

- `outputs/tables/trend_summary_1990_2023.csv`
- `outputs/tables/year_over_year_percent_change.csv`
- `outputs/tables/time_series_diagnostics.csv`
- `outputs/tables/stationarity_tests_by_d.csv`
- `outputs/tables/recommended_d_by_series.csv`
- `outputs/tables/primary_age_standardized_decomposition_period5.csv`

Forecast tables:

- `outputs/tables/classical_model_evaluation_2018_2023.csv`
- `outputs/tables/classical_forecasts_2024_2030.csv`
- `outputs/tables/all_model_evaluation_2018_2023.csv`
- `outputs/tables/best_models_by_series.csv`
- `outputs/tables/best_model_forecasts_2024_2030.csv`
- `outputs/tables/best_model_forecasts_2024_2030_with_uncertainty.csv`
- `outputs/tables/all_model_forecasts_2024_2030.csv`
- `outputs/model_outputs/all_model_test_predictions_2018_2023.csv`
- `outputs/forecasts/best_model_forecasts_2024_2030.csv`
- `outputs/forecasts/best_model_forecasts_2024_2030_with_uncertainty.csv`
- `outputs/forecasts/forecast_uncertainty_simulations_2024_2030.csv`
- `outputs/forecasts/all_model_forecasts_2024_2030.csv`

Figures:

- `outputs/figures/trend/incidence_rates_trend.png`
- `outputs/figures/trend/mortality_rates_trend.png`
- `outputs/figures/trend/incidence_mortality_numbers_trend.png`
- `outputs/figures/eda/primary_age_standardized_series_over_time.png`
- `outputs/figures/eda/primary_age_standardized_rates_by_sex.png`
- `outputs/figures/eda/primary_age_standardized_incidence_mortality_bar_comparison.png`
- `outputs/figures/eda/mortality_to_incidence_ratio_trend.png`
- `outputs/figures/eda/acf_pacf_incidence_age_standardized_rate.png`
- `outputs/figures/eda/acf_pacf_mortality_age_standardized_rate.png`
- `outputs/figures/eda/decomposition_incidence_age_standardized_rate_period5.png`
- `outputs/figures/eda/decomposition_mortality_age_standardized_rate_period5.png`
- `outputs/figures/forecasts/classical_forecasts_2024_2030_subplots.png`
- `outputs/figures/evaluation/all_model_test_predictions_subplots.png`
- `outputs/figures/evaluation/evaluation_errors_classical_models.png`
- `outputs/figures/evaluation/evaluation_errors_deep_learning_models.png`
- `outputs/figures/evaluation/evaluation_errors_all_models.png`
- `outputs/figures/forecasts/best_model_forecasts_2024_2030_subplots.png`
- `outputs/figures/uncertainty/best_model_forecasts_2024_2030_with_uncertainty.png`
- `outputs/figures/forecasts/all_model_forecasts_incidence_age_standardized_rate_2x4.png`
- `outputs/figures/forecasts/all_model_forecasts_mortality_age_standardized_rate_2x4.png`
- `outputs/figures/forecasts/all_model_forecasts_mortality_to_incidence_age_standardized_ratio_2x4.png`
- `outputs/figures/forecasts/mortality_to_incidence_ratio_forecasts_by_model_2x4.png`
- `outputs/figures/eda/*.png`

## Stationarity Handling

Stationarity is handled by model type:

- `scripts/00_eda_stationarity.py` runs ADF and KPSS tests for `d = 0, 1, 2`
  and saves the recommended differencing order for each modeling series.
- ARIMA selects `d` from ADF/KPSS evidence on the training series, then searches
  low-order `(p,d,q)` candidates with that selected `d`.
- LSTM and multistep LSTM use a first-difference transform before scaling and
  training. Forecasted differences are inverted back to the original TB scale
  by cumulative summation from the last observed value.
- ARIMA+LSTM uses the same ARIMA `d` selection, then trains the LSTM on ARIMA
  residuals.
- ETS and Drift are retained as nonstationary trend baselines because they model
  level/trend structure directly.
- Stationarity diagnostics are saved in `outputs/tables/stationarity_tests_by_d.csv`
  and `outputs/tables/recommended_d_by_series.csv`.

## GitHub Preparation

The repository is ready to publish with code, documentation, processed outputs,
and figures. Before pushing, review whether you want to include the raw GBD CSV
in `data/TB_df/`; it is public IHME output but still larger than the code files.

Recommended first commit:

```bash
git init
git add .
git status
git commit -m "Initial Ghana TB forecasting pipeline"
```

Then create an empty GitHub repository and push:

```bash
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```
