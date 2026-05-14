# Methods Documentation: Ghana TB Trend and Forecasting Models

## Study Aim

This project analyzes tuberculosis (TB) burden trends in Ghana using Global
Burden of Disease 2023 data from 1990 to 2023, then forecasts TB burden from
2024 to 2030.

The main analysis focuses on three age-standardized national endpoints:

- TB incidence age-standardized rate
- TB mortality age-standardized rate
- Mortality-to-incidence ratio, computed as the mortality age-standardized rate
  divided by the incidence age-standardized rate

All-age rates and counts are retained as optional secondary analyses. The
age-standardized endpoints are preferred because they reduce distortion from
changes in Ghana's age structure over time and are more appropriate for
comparing disease risk across years. The ratio is included as a derived
indicator of mortality burden relative to incident disease burden; it should be
interpreted as a population-level mortality-to-incidence ratio, not an
individual-level case fatality ratio.

## Data Source

The data were downloaded from the IHME GBD Results Tool:

> Global Burden of Disease Collaborative Network. Global Burden of Disease Study
> 2023 (GBD 2023) Results. Seattle, United States: Institute for Health Metrics
> and Evaluation (IHME), 2024. Available from
> https://vizhub.healthdata.org/gbd-results/.

The raw data file is:

```text
data/TB_df/IHME-GBD_2023_DATA-42460a02-1.csv
```

The prepared modeling table is:

```text
data/processed/ghana_tb_modeling_wide.csv
```

## Analysis Targets

The primary target series are annual univariate time series:

```text
y_t, t = 1990, 1991, ..., 2023
```

where `y_t` is either:

- incidence age-standardized rate, or
- mortality age-standardized rate, or
- mortality-to-incidence ratio:

```text
MIR_t = mortality_age_standardized_rate_t / incidence_age_standardized_rate_t
```

Forecasts are produced for:

```text
t = 2024, 2025, ..., 2030
```

## Train/Test Design

Models are evaluated using a temporal holdout:

- Training period: 1990-2017
- Test period: 2018-2023
- Final forecast period: 2024-2030

This preserves the time order of the data. The model is fit on historical
observations only and evaluated on later observations.

## Performance Metrics

For observed values `y_t` and forecasts `\hat{y}_t`, the following metrics are
computed:

Mean absolute error:

```text
MAE = (1 / n) * sum(|y_t - yhat_t|)
```

Root mean squared error:

```text
RMSE = sqrt((1 / n) * sum((y_t - yhat_t)^2))
```

Mean absolute percentage error:

```text
MAPE = (100 / n) * sum(|(y_t - yhat_t) / y_t|)
```

RMSE is used for primary model ranking because it penalizes larger forecast
errors more strongly. MAE and MAPE are reported for interpretability.

## Stationarity Assessment

Stationarity is assessed using both the Augmented Dickey-Fuller (ADF) test and
the KPSS test.

The ADF test has a unit-root null hypothesis:

```text
H0: the series has a unit root
H1: the series is stationary
```

The KPSS test uses the opposite direction:

```text
H0: the series is stationary
H1: the series is not stationary
```

Using both tests gives a more cautious assessment than using either test alone.
For each primary target series, stationarity is checked at differencing orders:

```text
d = 0, 1, 2
```

The smallest `d` supported by both tests is preferred. If ADF and KPSS do not
agree, the smallest `d` supported by at least one test is used as a pragmatic
fallback for the short annual series. The full-period diagnostic tables are
reported for the three primary modeling series. During model evaluation and
final forecasting, ARIMA-type models repeat this selection rule before each fit
using only the data available to that fit, then pass the selected `d`
explicitly into the ARIMA fitting function.

Differencing is defined as:

```text
Delta y_t = y_t - y_(t-1)
```

and repeated differencing is:

```text
Delta^d y_t
```

Stationarity outputs are saved to:

```text
outputs/tables/stationarity_tests_by_d.csv
outputs/tables/recommended_d_by_series.csv
```

## Models Added

The full model competition now includes eight models:

1. Naive
2. Drift
3. ARIMA
4. ETS
5. LSTM
6. Causal TCN
7. ARIMA-LSTM
8. Multistep LSTM

The newest method added is:

```text
Causal_TCN
```

It replaces the earlier directional CNN idea with a more appropriate temporal
convolutional architecture for ordered sequence forecasting.

## 1. Naive Forecast

The naive model forecasts all future values as the last observed value:

```text
yhat_(T+h) = y_T
```

where `T` is the final training year and `h` is the forecast horizon.

Why it is included:

- It is a minimal benchmark.
- Any useful model should generally outperform it.
- It is often competitive when changes are slow.

For smooth GBD time series, the naive model can perform reasonably over short
forecast horizons, but it cannot extrapolate systematic decline.

## 2. Drift Model

The drift model extends a straight-line trend from the first to the last
training observation:

```text
yhat_(T+h) = y_T + h * ((y_T - y_1) / (T - 1))
```

Why it works here:

- The age-standardized TB incidence and mortality rates decline smoothly.
- The model has very few parameters.
- It captures long-run monotonic trend without overfitting annual noise.

In the current primary analysis, Drift is the best model for both
age-standardized incidence and mortality rates.

## 3. ARIMA

An ARIMA model is written as:

```text
ARIMA(p, d, q)
```

where:

- `p` is the autoregressive order,
- `d` is the differencing order,
- `q` is the moving-average order.

After differencing `d` times, the model is:

```text
phi(B) Delta^d y_t = c + theta(B) epsilon_t
```

where:

- `B` is the backshift operator,
- `phi(B)` is the autoregressive polynomial,
- `theta(B)` is the moving-average polynomial,
- `epsilon_t` is white-noise error.

Implementation details:

- `d` is selected from ADF/KPSS evidence before each ARIMA fit using the data
  available to that fit: the training series during holdout evaluation and the
  full observed series during final forecasting.
- Low-order `(p,d,q)` candidates are compared using AIC.
- Forecasts are generated from the best AIC model.

Why it may work here:

- ARIMA is designed for short univariate time series.
- Differencing can remove trend/nonstationarity.
- Autoregressive and moving-average terms can capture serial dependence.

Why it may not win here:

- The primary series are very smooth and short.
- The holdout period is only six years.
- Higher-order ARIMA models may overfit.

## 4. ETS

ETS refers to exponential smoothing state-space models. The current
implementation compares:

- no trend,
- additive trend,
- damped additive trend.

A simple additive trend ETS model can be written as:

```text
Level:  l_t = alpha y_t + (1 - alpha)(l_(t-1) + b_(t-1))
Trend:  b_t = beta(l_t - l_(t-1)) + (1 - beta)b_(t-1)
Forecast: yhat_(t+h) = l_t + h b_t
```

A damped trend modifies the forecast path so the trend gradually weakens:

```text
yhat_(t+h) = l_t + (phi + phi^2 + ... + phi^h)b_t
```

where `0 < phi < 1`.

Why it is included:

- ETS is appropriate for level/trend time series.
- It does not require the same stationarity assumptions as ARIMA.
- Damped trend is useful when long-term linear extrapolation may be too strong.

## Deep Learning Data Framing

The neural models do not receive the raw year column as an input feature.
Instead, each annual series is converted into supervised lag-window examples.
This keeps the task purely univariate and avoids leakage from future years.

For a raw level series:

```text
y_1990, y_1991, ..., y_2023
```

the first-differenced series is:

```text
d_1991 = y_1991 - y_1990
d_1992 = y_1992 - y_1991
...
d_2023 = y_2023 - y_2022
```

The neural models are trained on the differenced series:

```text
d_t = Delta y_t
```

The differences are standardized using the training period only:

```text
z_t = (d_t - mean_train) / sd_train
```

where `z_t` is the standardized differenced value used by the neural network.

### Lag-Window Construction

For a lag length `k = 5`, one-step supervised examples are constructed as:

```text
Input X_i  = [z_i, z_(i+1), z_(i+2), z_(i+3), z_(i+4)]
Target y_i = z_(i+5)
```

The input tensor shape for the one-step LSTM and Causal TCN is:

```text
samples x time_steps x features
```

In the current scripts:

```text
features = 1
time_steps = seq_len
```

so a batch has shape:

```text
n_samples x seq_len x 1
```

### Concrete Example

Suppose the age-standardized incidence rate has the following simplified values:

```text
Year:  1990   1991   1992   1993   1994   1995   1996
Rate:  485.2  464.7  445.3  427.1  410.8  395.8  382.0
```

The first differences are:

```text
1991: -20.5
1992: -19.4
1993: -18.2
1994: -16.3
1995: -15.0
1996: -13.8
```

With `seq_len = 5`, the first one-step neural training sample is:

```text
Input:  [-20.5, -19.4, -18.2, -16.3, -15.0]
Target: -13.8
```

After standardization, the network sees the scaled version of those values:

```text
Input tensor shape:  1 x 5 x 1
Target shape:        1
```

This means the network learns a mapping from the previous five annual changes
to the next annual change.

### Recursive Forecasting

For recursive one-step models, the model predicts one future difference at a
time:

```text
zhat_(T+1) = f(z_(T-k+1), ..., z_T)
zhat_(T+2) = f(z_(T-k+2), ..., z_T, zhat_(T+1))
...
```

The predicted standardized differences are inverse transformed:

```text
dhat_(T+h) = zhat_(T+h) * sd_train + mean_train
```

Then converted back to levels:

```text
yhat_(T+h) = y_T + sum_{j=1}^{h} dhat_(T+j)
```

This is how the one-step LSTM and Causal TCN produce 2024-2030 forecasts.

### Direct Multistep Forecasting

The multistep LSTM uses the same input window but predicts the full future
horizon at once. For horizon `H = 6` in the 2018-2023 holdout:

```text
Input X_i = [z_i, z_(i+1), z_(i+2), z_(i+3), z_(i+4)]
Target Y_i = [z_(i+5), z_(i+6), z_(i+7), z_(i+8), z_(i+9), z_(i+10)]
```

The target tensor shape is:

```text
n_samples x horizon
```

For final 2024-2030 forecasting, the horizon is 7:

```text
2024, 2025, 2026, 2027, 2028, 2029, 2030
```

The network therefore directly outputs seven future annual changes, which are
then inverse transformed and accumulated back to the rate scale.

## 5. LSTM

Long short-term memory (LSTM) is a recurrent neural network designed to learn
temporal dependencies through gated memory.

A simplified LSTM cell uses gates:

```text
i_t = sigmoid(W_i x_t + U_i h_(t-1) + b_i)
f_t = sigmoid(W_f x_t + U_f h_(t-1) + b_f)
o_t = sigmoid(W_o x_t + U_o h_(t-1) + b_o)
g_t = tanh(W_g x_t + U_g h_(t-1) + b_g)
c_t = f_t * c_(t-1) + i_t * g_t
h_t = o_t * tanh(c_t)
```

Implementation details:

- The model is trained on first differences, not raw levels.
- Inputs are backward-looking lag windows.
- One-step forecasts are generated recursively.
- Predicted differences are inverted back to the original rate scale.
- Training uses TimeSeriesSplit cross-validation to preserve temporal order.
- Each fold uses Adam with learning rate 0.001, batch size 32, and early stopping with patience 10.
- The final model is then refit on all available training windows before recursive forecasting.

Data entry into the model:

```text
Input X shape:  n_samples x seq_len x 1
Target y shape: n_samples
```

Each input sample contains the previous `seq_len` standardized annual changes.
The target is the next standardized annual change.

For example, with `seq_len = 5`:

```text
X = [Delta y_2012, Delta y_2013, Delta y_2014, Delta y_2015, Delta y_2016]
y =  Delta y_2017
```

The LSTM reads the sequence in historical order and returns a final hidden state
from the last time step. A linear layer maps that hidden state to the next
differenced value:

```text
zhat_(t+1) = W h_t + b
```

Why it is included:

- It is a widely used neural time-series model.
- It can learn nonlinear lag relationships.

Why caution is needed:

- The dataset has only 34 annual observations.
- Neural models can overfit small, smooth public-health time series.
- Performance should be interpreted relative to simple baselines.

## 6. Causal TCN

The Causal Temporal Convolutional Network (Causal TCN) is the additional model
added to replace the earlier directional CNN idea.

A causal convolution predicts each output using only current and past inputs.
For a kernel of size `K` and dilation `r`:

```text
z_t = sum_{k=0}^{K-1} w_k x_(t - r*k)
```

The model uses left padding only, so future observations cannot enter the
prediction. Dilated convolutions allow the model to see a wider historical
window without requiring many layers.

The implemented TCN uses:

- first-differenced input series,
- causal one-dimensional convolutions,
- dilation rates `(1, 2)`,
- residual TCN blocks,
- recursive one-step forecasting,
- inverse transformation back to the original rate scale.
- Training follows the same TimeSeriesSplit cross-validation and final refit
  protocol as the LSTM, with Adam, learning rate 0.001, batch size 32, and
  early stopping patience 10.

Data entry into the model:

```text
Input X shape before convolution: n_samples x seq_len x 1
Input X shape inside Conv1D:      n_samples x 1 x seq_len
Target y shape:                   n_samples
```

The tensor is transposed before convolution because PyTorch `Conv1d` expects:

```text
batch x channels x sequence_length
```

For a sample:

```text
X = [Delta y_2012, Delta y_2013, Delta y_2014, Delta y_2015, Delta y_2016]
```

the causal TCN predicts:

```text
y = Delta y_2017
```

The causal convolution is left-padded only. Therefore, even inside the
convolutional layers, the representation at a time point can only use present
and past values from the input window. The final time step representation is
used for the one-step forecast.

Why it is more appropriate than a plain directional CNN:

- A plain CNN can accidentally behave like a generic pattern recognizer without
  a clear temporal forecasting structure.
- A causal TCN explicitly respects time direction.
- Dilations increase the receptive field while keeping the model compact.
- Residual blocks improve optimization stability.

Why it may work here:

- It is smaller and often easier to train than recurrent models.
- It can learn local changes in the slope of the differenced series.

Why caution is needed:

- The annual dataset is short.
- The dominant pattern is smooth decline, which simple trend models already
  capture well.

## 7. ARIMA-LSTM Hybrid

The ARIMA-LSTM hybrid decomposes the forecast problem into a linear component
and a residual nonlinear component.

First, ARIMA is fit:

```text
y_t = yhat_t^ARIMA + e_t
```

Then an LSTM is trained on the ARIMA residuals:

```text
e_t = f_LSTM(e_(t-1), e_(t-2), ..., e_(t-k)) + eta_t
```

The final forecast is:

```text
yhat_(T+h) = yhat_(T+h)^ARIMA + ehat_(T+h)^LSTM
```

Training follows the same TimeSeriesSplit cross-validation and final refit
protocol as the LSTM, with Adam, learning rate 0.001, batch size 32, and
early stopping patience 10.

Data entry into the residual LSTM:

```text
Input X = [e_(t-k), ..., e_(t-1)]
Target y = e_t
```

The residuals are standardized before LSTM training:

```text
z_t^e = (e_t - mean_residual_train) / sd_residual_train
```

At forecast time, ARIMA first produces a baseline forecast path. The residual
LSTM recursively predicts future residual corrections. The final forecast adds
the two components:

```text
Final forecast = ARIMA forecast + LSTM residual forecast
```

Why it is included:

- ARIMA handles linear autocorrelation and differencing.
- LSTM can model nonlinear residual structure if present.

Why caution is needed:

- Residual learning is difficult with only 34 observations.
- If ARIMA already captures most structure, residual neural modeling may add
  noise rather than signal.

## 8. Multistep LSTM

The multistep LSTM directly predicts the full forecast horizon rather than
recursively predicting one year at a time.

For a lag window of length `k` and horizon `H`:

```text
[Delta yhat_(T+1), ..., Delta yhat_(T+H)] =
f_LSTM(Delta y_(T-k+1), ..., Delta y_T)
```

Predicted differences are then accumulated from the final observed level:

```text
yhat_(T+h) = y_T + sum_{j=1}^{h} Delta yhat_(T+j)
```

Data entry into the model:

```text
Input X shape:  n_samples x seq_len x 1
Target Y shape: n_samples x horizon
```

Example with `seq_len = 5` and a 6-year test horizon:

```text
Input:
[Delta y_2007, Delta y_2008, Delta y_2009, Delta y_2010, Delta y_2011]

Target:
[Delta y_2012, Delta y_2013, Delta y_2014, Delta y_2015, Delta y_2016, Delta y_2017]
```

For final 2024-2030 forecasting, the model outputs:

```text
[Delta yhat_2024, Delta yhat_2025, ..., Delta yhat_2030]
```

Why it is included:

- It avoids recursive error accumulation.
- It explicitly learns the joint 2024-2030 forecast path.

Why caution is needed:

- Direct multistep learning requires enough examples of full horizons.
- With annual 1990-2023 data, the number of supervised windows is small.
- The implementation still uses the same TimeSeriesSplit cross-validation and
  final refit protocol, but the forecast is direct rather than recursive.

## Current Primary Results

Using the 2018-2023 holdout period and the three primary age-standardized
series, the best models by RMSE are currently:

```text
Incidence age-standardized rate: Drift
Mortality age-standardized rate: Drift
Mortality-to-incidence ratio: Causal TCN
```

This is methodologically plausible because the incidence and mortality GBD
series show smooth, monotonic long-run decline, while the derived ratio can
contain subtler changes in the relationship between death burden and incident
disease burden. The result should not be interpreted as a broad victory of
neural models over classical models. Instead, it suggests that parsimonious
trend models are favored for the smoother rate endpoints, while the causal TCN
may capture short-lag nonlinear changes in the derived ratio.

## Recommended Methods Text

Suggested manuscript wording:

> We evaluated eight forecasting models for annual TB incidence
> age-standardized rate, mortality age-standardized rate, and the derived
> mortality-to-incidence ratio in Ghana: naive persistence, drift, ARIMA,
> exponential smoothing, LSTM, causal temporal convolutional network (TCN),
> ARIMA-LSTM hybrid, and direct multistep LSTM. Models were trained on 1990-2017
> data and evaluated on a temporal holdout period from 2018-2023 using MAE,
> RMSE, and MAPE. Stationarity was assessed using ADF and KPSS tests across
> differencing orders d = 0, 1, and 2. ARIMA differencing order was selected
> before each model fit using the same ADF/KPSS rule, and low-order ARIMA
> candidates were then compared by AIC.
> Neural models were trained on first-differenced series and forecasts were
> inverted to the original scale by cumulative summation from the final observed
> value. The best model by RMSE was refit on the full 1990-2023 series and used
> to forecast 2024-2030. Forecast uncertainty intervals were estimated by
> simulating historical trajectories from the GBD lower, point, and upper
> estimates, refitting the selected model to each trajectory, and summarizing
> the 2.5th and 97.5th percentiles of the simulated forecasts. For the
> mortality-to-incidence ratio, the uncertainty simulation is applied directly
> to the ratio series rather than recomputing it from separate incidence and
> mortality forecasts. The outputs include both a GBD-input-only interval and a
> wider combined interval that adds centered holdout residual error.

## Important Limitations

- The series are annual and short: only 34 observations.
- Deep learning models are included as comparators, but the data volume is small
  for neural networks.
- Forecast uncertainty intervals propagate GBD input uncertainty through the
  selected best models, but they do not fully capture structural model
  uncertainty, omitted covariates, or unexpected future shocks.
- The uncertainty tables and figures report two interval types: a GBD-input
  interval and a combined interval that adds empirical holdout error.
- National-level univariate models do not include covariates such as HIV
  prevalence, treatment coverage, socioeconomic status, diagnostics, or TB
  program indicators.
- Forecasts assume continuity of historical patterns and do not explicitly model
  future policy shocks, outbreaks, economic disruption, or diagnostic changes.

## Citations

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of
  generic convolutional and recurrent networks for sequence modeling. arXiv:
  1803.01271. https://arxiv.org/abs/1803.01271
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time
  Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for
  autoregressive time series with a unit root. *Journal of the American
  Statistical Association*, 74(366a), 427-431.
  https://doi.org/10.1080/01621459.1979.10482531
- Global Burden of Disease Collaborative Network. (2024). *Global Burden of
  Disease Study 2023 (GBD 2023) Results*. Institute for Health Metrics and
  Evaluation. https://vizhub.healthdata.org/gbd-results/
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
  Computation*, 9(8), 1735-1780.
  https://doi.org/10.1162/neco.1997.9.8.1735
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
- Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: The
  forecast package for R. *Journal of Statistical Software*, 27(3), 1-22.
  https://www.jstatsoft.org/v27/i03/
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing
  the null hypothesis of stationarity against the alternative of a unit root.
  *Journal of Econometrics*, 54(1-3), 159-178.
