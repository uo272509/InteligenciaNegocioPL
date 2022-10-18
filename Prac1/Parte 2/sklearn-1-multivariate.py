import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np

boston_dataset = datasets.load_boston()
print(boston_dataset.feature_names)

X_full = boston_dataset.data
Y = boston_dataset.target
print(X_full.shape)
print(Y.shape)
X = X_full[:, :-1]

plt.title("Variable target ordenada")
orden = np.argsort(Y)
horizontal = np.arange(Y.shape[0])

kernel = "precomputed"

regressor = LinearRegression(normalize=True)
regressor.fit(X, Y)
plt.title("SVR con " + kernel + " en lugar de rbf")
plt.scatter(horizontal, Y[orden], color='black', linewidth=3, label="Valores")
plt.scatter(horizontal, regressor.predict(X)[orden], color='red', label="Regresión Lineal")

regressor = SVR(kernel=kernel, C=1e1, epsilon=1)
regressor.fit(X, Y)
plt.scatter(horizontal, regressor.predict(X)[orden], color='green', label="SVR con " + kernel)

regressor = RandomForestRegressor()
regressor.fit(X, Y)
plt.scatter(horizontal, regressor.predict(X)[orden], color='blue', label="RandomForesRegressor")
plt.legend()
plt.show()
