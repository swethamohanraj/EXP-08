# EXP - 8 IMPLEMENTATION OF ARIMA MODEL

## AIM:

Implementation of ARIMA model using python.

## ALGORITHM:

1) Explore the dataset
2) Check for stationarity of time series
    * time series plot
    * ACF plot and PACF plot
    * ADF test
    * Transform to stationary: differencing
3) Determine ARIMA models parameters p, q
4) Fit the ARIMA model
5) Make time series predictions
    * Auto-fit the ARIMA model
7) Evaluate model predictions

## PROGRAM:

```python

import pandas as pd
df = pd.read_csv('website_data.csv')
df.info()
df.plot()
import numpy as np
df = np.log(df) # don't forget to transform the data back when making real predictions
df.plot()
msk = (df.index < len(df)-30)
df_train = df[msk].copy()
df_test = df[~msk].copy()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

acf_original = plot_acf(df_train)

pacf_original = plot_pacf(df_train)
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(df_train)
print(f'p-value: {adf_test[1]}')
df_train_diff = df_train.diff().dropna()
df_train_diff.plot()
acf_diff = plot_acf(df_train_diff)

pacf_diff = plot_pacf(df_train_diff)
adf_test = adfuller(df_train_diff)
print(f'p-value: {adf_test[1]}')
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df_train, order=(2,1,0))
model_fit = model.fit()
print(model_fit.summary())
import matplotlib.pyplot as plt
residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(title='Density', kind='kde', ax=ax[1])
plt.show()
acf_res = plot_acf(residuals)

pacf_res = plot_pacf(residuals)
forecast_test = model_fit.forecast(len(df_test))

df['forecast_manual'] = [None]*len(df_train) + list(forecast_test)

df.plot()
import pmdarima as pm
auto_arima = pm.auto_arima(df_train, stepwise=False, seasonal=False)
auto_arima
auto_arima.summary()
forecast_test_auto = auto_arima.predict(n_periods=len(df_test))
df['forecast_auto'] = [None]*len(df_train) + list(forecast_test_auto)

df.plot()

```
## OUTPUT:

### ACF - PACF

<img width="341" alt="image" src="https://github.com/Monisha-11/IMPLEMENTATION-OF-ARIMA-MODEL/assets/93427240/a17dcca4-136e-41ab-8292-bb263f3cf869">

### TRAIN_DIFF

<img width="352" alt="image" src="https://github.com/Monisha-11/IMPLEMENTATION-OF-ARIMA-MODEL/assets/93427240/8fcb099e-7433-4692-bf02-657693a47506">

### AFTER, PLOT THE TRAIN_DIFF

<img width="346" alt="image" src="https://github.com/Monisha-11/IMPLEMENTATION-OF-ARIMA-MODEL/assets/93427240/b9902297-b204-4b61-9e28-9ea6819beedf">

### RESIDUALS

<img width="347" alt="image" src="https://github.com/Monisha-11/IMPLEMENTATION-OF-ARIMA-MODEL/assets/93427240/52b97011-4df0-4eb1-a4f9-ab1438fa4391">

### AFTER, PLOT THE RESIDUALS

<img width="351" alt="image" src="https://github.com/Monisha-11/IMPLEMENTATION-OF-ARIMA-MODEL/assets/93427240/f6fa10e2-0c0b-441e-9956-f3cfb40936f9">

### FINIAL PREDICTION

<img width="338" alt="image" src="https://github.com/Monisha-11/IMPLEMENTATION-OF-ARIMA-MODEL/assets/93427240/6f8675f8-fe69-4eaa-ace2-df636b23c422">


## RESULT:

Thus the program run successfully based on the ARIMA model.**
