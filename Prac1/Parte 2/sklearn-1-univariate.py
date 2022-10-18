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
N = Y.shape[0]

# Se elige la variable mas dependiente de la salida
# SE ESTUDIARA EN CLASE DE TEORIA MAS ADELANTE
# k es el número de variables que se eligen
selector = SelectKBest(f_regression, k=1)
selector.fit(X_full, Y)
X = X_full[:, selector.get_support()]
print(X.shape)
plt.subplot(421, aspect=1)
plt.title("Variable más dependiente de la salida")
plt.scatter(X, Y, color='black')

plt.subplot(422, aspect=4)
plt.title("Variable más dependiente de la salida ordenada por precios")
idx = np.argsort(Y)
plt.scatter(range(N), Y[idx], color='black')

plt.subplot(423, aspect=1)
plt.title("Estimación lineal sobre la variable")
regressor = LinearRegression(normalize=True)
regressor.fit(X, Y)
plt.scatter(X, Y, color='black')
plt.plot(X, regressor.predict(X), color='blue', linewidth=3)

plt.subplot(424, aspect=4)
plt.title("Estimación lineal sobre la variable ordenada por precios")
plt.scatter(range(N), Y[idx], color='black')
plt.scatter(range(N), regressor.predict(X)[idx], color='blue')

plt.subplot(425, aspect=1)
plt.title("SVR (Support Vector Regression) sobre la variable")
regressor = SVR(kernel='rbf', C=1e1, epsilon=1)
regressor.fit(X, Y)
plt.scatter(X, Y, color='black')
plt.scatter(X, regressor.predict(X), color='blue', linewidth=3)

plt.subplot(426, aspect=4)
plt.title("SVR (Support Vector Regression) sobre la variable ordenada")
plt.scatter(range(N), Y[idx], color='black')
plt.scatter(range(N), regressor.predict(X)[idx], color='blue')

plt.subplot(427, aspect=1)
plt.title("Random Forest Regression sobre la variable")
regressor = RandomForestRegressor()
regressor.fit(X, Y)
plt.scatter(X, Y, color='black')
plt.scatter(X, regressor.predict(X), color='blue', linewidth=3)

plt.subplot(428, aspect=4)
plt.title("Random Forest Regression sobre la variable ordenada")
plt.scatter(range(N), Y[idx], color='black')
plt.scatter(range(N), regressor.predict(X)[idx], color='blue')
plt.show()
