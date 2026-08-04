import numpy as np
from statsmodels.tsa.arima.model import ARIMA

y_train = np.arange(100) + np.random.normal(0, 1, 100)
y_test = np.arange(100, 110) + np.random.normal(0, 1, 10)

model = ARIMA(y_train, order=(1, 1, 1))
fit_res = model.fit()

# append
test_res = fit_res.append(y_test, refit=False)
preds = test_res.fittedvalues[-len(y_test):]

print("Actual test:", y_test)
print("Preds append:", preds)
