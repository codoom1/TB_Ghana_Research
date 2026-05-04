# Methods

## Study Design and Data Source

This study used an ecological time-series forecasting design to characterize
historical trends and project future tuberculosis (TB) burden in Ghana. Annual
estimates were obtained from the Global Burden of Disease Study 2023 (GBD 2023)
Results Tool, produced by the Institute for Health Metrics and Evaluation
(IHME). The study period for observed data was 1990 to 2023, and forecasts were
generated for 2024 to 2030.

The source data were restricted to Ghana as the location and tuberculosis as the
cause. The main analysis used national estimates for both sexes combined. The
primary modeled outcomes were the age-standardized TB incidence rate, the
age-standardized TB mortality rate, and a derived mortality-to-incidence ratio.
All-age rates and all-age counts were retained as secondary analysis options in
the computational pipeline but were not prioritized for the main model
comparison because age-standardized rates are less affected by changes in the
population age structure over time.

The GBD 2023 data citation is:

Global Burden of Disease Collaborative Network. Global Burden of Disease Study
2023 (GBD 2023) Results. Seattle, United States: Institute for Health Metrics
and Evaluation (IHME), 2024. Available from
https://vizhub.healthdata.org/gbd-results/.

## Outcome Definitions

Let `t` denote calendar year, where `t = 1990, 1991, ..., 2023` for observed
data. The primary annual series were:

1. TB incidence age-standardized rate per 100,000 population.
2. TB mortality age-standardized rate per 100,000 population.
3. Mortality-to-incidence ratio, defined as:

```text
MIR_t = mortality_age_standardized_rate_t / incidence_age_standardized_rate_t
```

The mortality-to-incidence ratio was included as a derived indicator of
mortality burden relative to incident disease burden. Because it is computed
from population-level rates, it should be interpreted as a population-level
mortality-to-incidence ratio rather than an individual-level case fatality
ratio.

For the derived ratio, uncertainty bounds were also computed descriptively using
the lower mortality bound divided by the upper incidence bound for the lower
ratio bound, and the upper mortality bound divided by the lower incidence bound
for the upper ratio bound:

```text
MIR_lower_t = mortality_lower_t / incidence_upper_t
MIR_upper_t = mortality_upper_t / incidence_lower_t
```

The forecasting models were fitted to the GBD point estimate (`val`) for each
outcome. Forecast uncertainty intervals were then generated in a separate
simulation step by propagating the GBD lower and upper uncertainty bounds
through the selected best models.

## Data Preprocessing

The raw GBD extract was imported as a long-format table containing measure,
location, sex, age, cause, metric, year, point estimate, and uncertainty bounds.
Rows were filtered to Ghana, tuberculosis, both sexes combined, and the target
combinations of measure, age group, and metric. The filtered records were then
pivoted into a wide annual modeling table, with one row per year and one column
per target series. The final observed modeling table contained 34 annual
observations from 1990 to 2023.

The primary modeling table included the following columns:

```text
incidence_age_standardized_rate
mortality_age_standardized_rate
mortality_to_incidence_age_standardized_ratio
```

Secondary columns included all-age incidence and mortality rates and counts.
The analysis verified complete annual coverage for 1990 to 2023 before model
fitting. No interpolation was required because each target series had one value
for every year in the study period.

## Exploratory Trend Analysis

Exploratory data analysis was conducted before model fitting. Annual trends were
visualized using line plots for the age-standardized incidence and mortality
rates, grouped bar plots comparing incidence and mortality by year, and a
mortality-to-incidence ratio trend plot. Additional stratified visualizations
were produced by sex and condensed age group to describe the broader
epidemiological pattern in the GBD data.

Autocorrelation function (ACF) and partial autocorrelation function (PACF) plots
were generated for the primary age-standardized rate series. These plots were
used descriptively to assess temporal dependence and to support interpretation
of autoregressive and moving-average structure. Because the data are annual, no
monthly or quarterly seasonality was assumed. A decomposition with a five-year
period was used only as an exploratory trend-cycle summary and was not treated
as evidence of formal seasonal periodicity.

## Stationarity Assessment

Stationarity was assessed for each target series using the Augmented
Dickey-Fuller (ADF) test and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.
The ADF test evaluates the null hypothesis that the series has a unit root,
whereas the KPSS test evaluates the null hypothesis that the series is
stationary. These tests are therefore complementary: evidence for stationarity
corresponds to a small ADF p-value and a large KPSS p-value.

To support the formal tests, each primary age-standardized series was also
plotted with its observed annual values, a trailing 5-year rolling mean, and a
5-year rolling standard deviation band. This visualization was used to inspect
whether the central tendency and variability changed over time, which is a
common visual sign of nonstationarity in short annual series.

For a series `y_t`, first differencing was defined as:

```text
Delta y_t = y_t - y_(t-1)
```

Repeated differencing was written as `Delta^d y_t`. Stationarity tests were
performed at differencing orders `d = 0, 1, 2`. The smallest differencing order
supported by both ADF and KPSS was preferred. If the two tests did not agree,
the smallest differencing order supported by at least one test was retained as a
pragmatic choice for short annual series. These diagnostics were used directly
for ARIMA-type models and descriptively for the other models. Specifically,
the ARIMA model and the ARIMA component of the ARIMA-LSTM hybrid selected the
differencing order from the ADF/KPSS diagnostics. The standalone neural-network
models used a fixed first-difference transformation as a preprocessing step to
reduce trend and place the learning task on annual changes.

This distinction was made because the differencing order `d` is a formal
parameter of ARIMA models, but not a native parameter of LSTM or TCN models.
ARIMA is specified as `ARIMA(p, d, q)`, so selecting `d` is part of the model
definition. In contrast, LSTM and TCN models estimate a nonlinear function from
lagged inputs to future outputs and do not automatically contain a differencing
order unless it is imposed during preprocessing. A fixed first-difference
transform was therefore used for the standalone neural models as a conservative
preprocessing choice. This reduced the dominance of the long-term downward
trend while preserving the annual change signal. Although second differencing
can satisfy formal stationarity tests for some outcomes, applying `d = 2` to a
34-year annual series can amplify noise and make neural models learn changes in
changes rather than interpretable epidemiological movement. First differencing
also permits stable inverse reconstruction of forecasts by cumulative summation
from the last observed level. For these reasons, data-driven differencing order
selection was reserved for ARIMA-based models, while neural-network models used
a consistent first-difference transformation.

## Forecasting Framework

All forecasting models were treated as univariate time-series models. For each
outcome, models were trained using annual observations from 1990 to 2017 and
evaluated on a temporal holdout period from 2018 to 2023. This validation
design preserved chronological ordering and avoided information leakage from
future years into model training. After evaluation, the best-performing model
for each outcome was refit using the full observed period, 1990 to 2023, and
used to forecast 2024 to 2030.

Eight competing models were evaluated:

1. Naive persistence model.
2. Drift model.
3. Autoregressive integrated moving average (ARIMA).
4. Exponential smoothing (ETS).
5. Long short-term memory neural network (LSTM).
6. Causal temporal convolutional network (causal TCN).
7. ARIMA-LSTM hybrid model.
8. Direct multistep LSTM.

The classical models were included to provide parsimonious benchmarks and
standard time-series comparators. The neural-network models were included to
evaluate whether nonlinear sequence models improved forecasting accuracy for
the observed TB burden series.

## Forecasting Models

### Naive Persistence Model

The naive model used the final observed training value as the forecast for all
future horizons. For horizon `h`, the forecast was:

```text
yhat_(T+h) = y_T
```

where `T` is the final year in the training period. This model was included as a
minimal benchmark because any useful forecasting model should generally perform
at least as well as persistence.

### Drift Model

The drift model extrapolated a linear trend from the first to the last training
observation:

```text
yhat_(T+h) = y_T + h * ((y_T - y_1) / (T - 1))
```

This model is appropriate as a simple comparator when disease burden changes
smoothly over time. It captures long-term monotonic decline or increase while
using very few parameters.

### ARIMA Model

ARIMA models were specified as `ARIMA(p, d, q)`, where `p` is the autoregressive
order, `d` is the differencing order, and `q` is the moving-average order. The
general model can be written as:

```text
phi(B) Delta^d y_t = c + theta(B) epsilon_t
```

where `B` is the backshift operator, `phi(B)` is the autoregressive polynomial,
`theta(B)` is the moving-average polynomial, and `epsilon_t` is a white-noise
error term. The differencing order `d` was selected using the ADF and KPSS
stationarity diagnostics described above. Low-order ARIMA candidates were then
compared using Akaike's information criterion (AIC), and the model with the
lowest AIC was retained for prediction.

### Exponential Smoothing Model

Exponential smoothing models forecast future values using weighted averages of
past observations, with larger weights assigned to more recent values. The
implementation considered level and trend components, including damped-trend
forms when supported by the data. In additive error and additive trend form, the
model may be summarized as:

```text
level_t = alpha y_t + (1 - alpha)(level_(t-1) + trend_(t-1))
trend_t = beta(level_t - level_(t-1)) + (1 - beta)trend_(t-1)
yhat_(t+h) = level_t + h trend_t
```

Damped-trend variants reduce the long-horizon influence of the trend term. ETS
models were included because they are well suited to short, smooth annual
series with slowly evolving levels and trends.

### LSTM Model

The LSTM model was used to learn nonlinear temporal dependence from lagged
values. Unlike ARIMA, the LSTM did not estimate the differencing order from the
stationarity tests. Instead, each series was transformed using a fixed
first-difference preprocessing step:

```text
z_t = y_t - y_(t-1)
```

The first-differenced series was converted into supervised learning samples
using a sliding window. For a sequence length of five, a training example had
the form:

```text
Input:  [z_(t-4), z_(t-3), z_(t-2), z_(t-1), z_t]
Target: z_(t+1)
```

The LSTM learned to predict the next annual change. Recursive one-step
forecasting was used for multi-year prediction: the model predicted the next
difference, appended that predicted difference to the input sequence, and then
predicted the following difference. Forecasted differences were transformed
back to the original scale by cumulative summation from the final observed
level:

```text
yhat_(T+h) = y_T + sum_{j=1}^{h} zhat_(T+j)
```

### Causal Temporal Convolutional Network

The causal temporal convolutional network (TCN) used one-dimensional causal
convolutions over lagged values of the fixed first-differenced series. Causal
convolution ensures that predictions for a future time point depend only on the
current and past values, not future values. Dilated convolutions were used to
increase the receptive field without requiring a large number of parameters.

Let the first-differenced input sequence be:

```text
z_1, z_2, ..., z_T
```

For supervised learning, the TCN received a lag window:

```text
X_t = [z_(t-s+1), z_(t-s+2), ..., z_t]
```

where `s` is the sequence length. A five-year lookback window was used
(`s = 5`) for the neural-network models. This choice was made for both
substantive and statistical reasons. Substantively, a five-year window provides
an interpretable summary of recent TB burden movement and is consistent with
the exploratory five-year trend-cycle decomposition used in the EDA. Statistically,
it is conservative for a 34-year annual series: it gives the neural models
recent temporal context while preserving more supervised training windows than
a longer lookback. Using the same five-year window across LSTM, Causal TCN, and
Multistep LSTM also makes the neural-model comparison more transparent. The
target was the next annual change:

```text
Target = z_(t+1)
```

In a standard noncausal convolution, the output at time `t` may be computed
using values on both sides of `t`, which can allow future information to enter
the prediction. In contrast, the causal TCN pads only on the left side of the
sequence and computes each feature using current and past inputs only. For a
kernel size `k` and dilation `d_l`, a single-channel causal convolutional
feature at time `t` can be written as:

```text
h_t = sum_{i=0}^{k-1} w_i x_(t - d_l * i)
```

where `w_i` is the convolution weight at kernel position `i`, `x_t` is the
input at time `t`, and `d_l` is the dilation factor for layer `l`. When
`d_l = 1`, the convolution uses adjacent lagged values. When `d_l > 1`, the
kernel skips over observations and can learn longer-range temporal structure:

```text
d_l = 1: x_t, x_(t-1), x_(t-2), ...
d_l = 2: x_t, x_(t-2), x_(t-4), ...
```

In this context, the dilation factor defines the spacing between lagged annual
observations included in the convolution. Larger dilation values allow the
model to incorporate information from farther back in time without increasing
the number of convolution weights.

The implementation used a small kernel and two dilation levels, `d_l = 1` and
`d_l = 2`, to capture short- and medium-range annual dependence while keeping
the model small enough for the 34-year series. With kernel size `k = 2` and
dilations `(1, 2)`, the stacked causal convolutions allow the model to combine
recent annual changes with changes from farther back in the lag window without
using future observations.

The receptive field describes how many past time points can influence a
prediction. For a stack of causal convolutional layers with kernel size `k` and
dilations `d_1, d_2, ..., d_L`, the approximate receptive field is:

```text
R = 1 + (k - 1) * sum_{l=1}^{L} d_l
```

Thus, increasing dilation expands the historical context without requiring a
large number of parameters. This is helpful for annual epidemiological data,
where the available sample size is small and highly parameterized neural
networks can overfit.

The TCN used residual blocks. If `F_l(.)` denotes the transformation learned by
the causal convolutional layers in block `l`, the residual output was:

```text
H_l = H_(l-1) + F_l(H_(l-1))
```

Residual connections help preserve the original lagged signal and stabilize
optimization. After the stacked causal residual blocks, the representation at
the final time step of the lag window was passed through a linear output layer
to predict the next first difference:

```text
zhat_(t+1) = beta_0 + beta' H_t
```

Multi-year forecasts were produced recursively. After predicting
`zhat_(t+1)`, the predicted value was appended to the input history and used to
predict `zhat_(t+2)`, continuing until the required forecast horizon was
reached. The predicted differences were then converted back to the original
outcome scale by cumulative summation from the last observed value.

The causal TCN was included as a more suitable sequence model than a generic
directional convolutional neural network because it explicitly respects time
ordering and avoids leakage from future values.

### ARIMA-LSTM Hybrid Model

The ARIMA-LSTM hybrid model decomposed forecasting into a linear component and a
nonlinear residual component. First, an ARIMA model was fitted to the training
series. The residuals were then computed as:

```text
e_t = y_t - yhat_ARIMA_t
```

An LSTM was trained on the residual sequence to learn remaining temporal
structure not captured by ARIMA. The final forecast combined both components:

```text
yhat_hybrid_(T+h) = yhat_ARIMA_(T+h) + ehat_LSTM_(T+h)
```

This hybrid approach was included because ARIMA can capture linear dependence
and differencing structure, while LSTM may capture nonlinear residual patterns.

### Direct Multistep LSTM

The direct multistep LSTM predicted the full forecast horizon jointly rather
than recursively predicting one year at a time. For a horizon `H`, the target
was:

```text
Target: [z_(t+1), z_(t+2), ..., z_(t+H)]
```

For the final forecast, `H = 7`, corresponding to 2024 through 2030. This
approach was included to reduce recursive error accumulation, although it
requires enough training windows to learn complete multi-year trajectories.

## Model Training and Validation

For each target outcome, the observed period was split into a training period
from 1990 to 2017 and a test period from 2018 to 2023. Each model was fit using
only the training period and then used to forecast the six holdout years. Model
accuracy was assessed by comparing the holdout forecasts with the observed GBD
estimates for 2018 to 2023.

The standalone neural-network models were trained on fixed first-differenced
series to reduce the effect of nonstationary level trends. This transformation
was not selected separately for each neural model; it was applied consistently
as a preprocessing choice because the annual national series were short and
trend-dominated. Input windows were generated from the training portion only.
Forecasts from neural models were converted back to the original outcome scale
using cumulative summation from the last observed training value. Negative
forecasts, if produced, were truncated to zero because incidence rates,
mortality rates, and ratios cannot be negative.

## Model Evaluation Metrics

Forecast accuracy was evaluated using mean absolute error (MAE), root mean
squared error (RMSE), and mean absolute percentage error (MAPE). For observed
values `y_t` and predicted values `yhat_t`, these metrics were defined as:

```text
MAE = (1 / n) sum |y_t - yhat_t|
```

```text
RMSE = sqrt((1 / n) sum (y_t - yhat_t)^2)
```

```text
MAPE = (100 / n) sum |(y_t - yhat_t) / y_t|
```

RMSE was used as the primary model-selection criterion because it penalizes
large forecast errors more strongly than MAE. MAE and MAPE were reported as
secondary metrics to support interpretation across outcomes with different
scales.

## Model Selection and Final Forecasting

For each outcome, models were ranked by RMSE on the 2018 to 2023 holdout
period. MAE and MAPE were used as secondary tie-breakers where needed. The
selected model for each outcome was then refit using all available observed
data from 1990 to 2023. Final forecasts were generated for 2024 through 2030.

All eight models were also refit on the full observed series to produce
model-specific forecast plots. These plots were used to compare alternative
forecast trajectories and to assess whether model-implied trends were
epidemiologically plausible.

Different endpoints were allowed to select different best-performing models
because the three target series have different temporal structures. The
age-standardized incidence and mortality rates are direct burden measures and
showed smooth long-term declines; therefore, a parsimonious linear trend model
could capture most of their forecast signal. The mortality-to-incidence ratio,
however, is a derived relative burden measure:

```text
MIR_t = mortality_rate_t / incidence_rate_t
```

This ratio reflects whether mortality is declining faster or slower than
incidence. As a result, it can contain local nonlinear changes even when both
underlying rates are declining. The causal TCN was therefore a plausible
candidate for this outcome because it models recent lagged annual changes
through causal and dilated convolutions. Its selection as the best model for
the ratio suggests that short-lag patterns in relative mortality burden were
more informative for this derived endpoint than a single long-run linear trend.
This endpoint-specific result was interpreted cautiously and was not taken as
evidence that the causal TCN was universally superior to classical models.

## Forecast Uncertainty Intervals

Forecast uncertainty was estimated using simulation to propagate uncertainty in
the historical GBD estimates through the selected best model for each endpoint.
For each year and outcome, the GBD point estimate, lower bound, and upper bound
were used to define a bounded beta-PERT distribution. The lower and upper GBD
bounds defined the distribution range, and the GBD point estimate was used as
the modal value. This choice was made because the GBD uncertainty intervals are
bounded and can be asymmetric around the point estimate.

For each simulation replicate, a complete historical trajectory from 1990 to
2023 was sampled from these year-specific distributions. The selected best
model for that endpoint was then refit to the simulated trajectory and used to
forecast 2024 to 2030. This procedure was repeated 100 times for each primary
endpoint. Forecast uncertainty intervals were summarized using the 2.5th and
97.5th percentiles of the simulated forecast distribution at each forecast
year:

```text
forecast_lower_t = percentile_2.5({yhat_t^(1), ..., yhat_t^(B)})
forecast_upper_t = percentile_97.5({yhat_t^(1), ..., yhat_t^(B)})
```

where `B` is the number of simulation replicates. These intervals should be
interpreted as simulation-based forecast uncertainty intervals that propagate
GBD input uncertainty through the selected model. They are not formal
frequentist confidence intervals and do not fully capture structural model
uncertainty, omitted covariates, or future epidemiological shocks.

A more technical description is as follows. The beta-PERT distribution is a parameterization of the beta distribution, defined by a lower bound $a$, an upper bound $b$, and a mode (most likely value) $m$ (where $a < m < b$). The shape parameters for the underlying beta distribution are:

$$
\alpha = 1 + \lambda \cdot \frac{m - a}{b - a}
$$

$$
\beta = 1 + \lambda \cdot \frac{b - m}{b - a}
$$

where $\lambda$ is a positive shape parameter (commonly $\lambda = 4$ for the standard PERT distribution). A random variable $X$ following the beta-PERT distribution is generated as:

$$
X = a + (b - a) \cdot Y
$$

where $Y \sim \mathrm{Beta}(\alpha, \beta)$. This construction ensures the distribution is bounded between $a$ and $b$, with the mode at $m$, and allows for asymmetry depending on the location of $m$. In this analysis, the GBD lower bound is $a$, the upper bound is $b$, and the GBD point estimate is $m$. The shape parameter $\lambda$ controls the peakedness of the distribution; higher values concentrate more probability near the mode.


## Software and Reproducibility

All analyses were conducted using Python. Data management and tabulation were
performed with pandas and NumPy. Statistical time-series models and
stationarity tests were implemented using statsmodels. Neural-network models
were implemented using PyTorch. Figures were produced using matplotlib and
seaborn. Progress reporting during model training was handled using tqdm.

The analysis pipeline was organized into separate scripts for data preparation,
exploratory analysis, model evaluation, and final forecasting. The individual
model classes were implemented in separate model scripts to improve
reproducibility and transparency. The main pipeline outputs included processed
modeling data, stationarity diagnostics, model-evaluation tables, test-period
predictions, final forecast tables, simulation-based uncertainty intervals, and
publication-oriented figures.

## Ethical Considerations

This study used publicly available, aggregate disease-burden estimates from the
GBD 2023 Results Tool. No individual-level or identifiable human participant
data were used. Therefore, institutional review board approval and informed
consent were not required.

## Methodological Limitations

Several limitations should be considered. First, the analysis used annual
national estimates, yielding only 34 observed time points. This restricts the
amount of information available for complex machine-learning models and
increases uncertainty in long-horizon forecasts. Second, the models were
univariate and did not incorporate potential predictors such as HIV prevalence,
TB treatment coverage, diagnostic expansion, socioeconomic indicators, or
health-system disruptions. Third, the uncertainty intervals propagated GBD
input uncertainty through the selected best models, but they did not fully
capture structural model uncertainty or unexpected future epidemiological
shocks. Fourth, the mortality-to-incidence ratio was derived from population
rates and should not be interpreted as a patient-level case fatality estimate.



## Result outline
1. raw Time seies plot

## References

1. Global Burden of Disease Collaborative Network. Global Burden of Disease
   Study 2023 (GBD 2023) Results. Seattle, United States: Institute for Health
   Metrics and Evaluation (IHME); 2024. Available from:
   https://vizhub.healthdata.org/gbd-results/.
2. Box GEP, Jenkins GM, Reinsel GC, Ljung GM. Time Series Analysis:
   Forecasting and Control. 5th ed. Hoboken: Wiley; 2015.
3. Hyndman RJ, Khandakar Y. Automatic time series forecasting: the forecast
   package for R. Journal of Statistical Software. 2008;27(3):1-22.
4. Dickey DA, Fuller WA. Distribution of the estimators for autoregressive time
   series with a unit root. Journal of the American Statistical Association.
   1979;74(366):427-431.
5. Kwiatkowski D, Phillips PCB, Schmidt P, Shin Y. Testing the null hypothesis
   of stationarity against the alternative of a unit root. Journal of
   Econometrics. 1992;54(1-3):159-178.
6. Hochreiter S, Schmidhuber J. Long short-term memory. Neural Computation.
   1997;9(8):1735-1780.
7. Bai S, Kolter JZ, Koltun V. An empirical evaluation of generic
   convolutional and recurrent networks for sequence modeling. arXiv.
   2018;1803.01271.
8. Zhang GP. Time series forecasting using a hybrid ARIMA and neural network
   model. Neurocomputing. 2003;50:159-175.
