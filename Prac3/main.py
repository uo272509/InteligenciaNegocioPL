from typing import Union

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile, f_regression, VarianceThreshold
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def mean_variance(df: Union[pd.DataFrame, np.ndarray]) -> float:
    """ Devuelve la varianza media de un dataframe o ndarray

    :param df: Dataframe o ndarray
    :return: varianza media
    """
    if type(df) == pd.DataFrame:
        return np.array([x.var() for name, x in df.items()]).mean()
    else:
        return np.array([x.var() for x in df.T]).mean()


def try_model(model, X: Union[pd.DataFrame, np.ndarray], Y: Union[pd.DataFrame, np.ndarray]) -> tuple[float, float, float]:
    """Ejecuta un modelo, imprime sus estadísticas por pantalla y devuelve esas estadísticas

    :param model: Modelo que usar para hacer la estimación
    :param X: Datos de entrada
    :param Y: Datos de salida
    :return: Error cuadrático medio, error medio y putuación de validación cruzada
    """
    model.fit(X, Y)
    prediction = model.predict(X)
    err = mean_squared_error(Y, prediction)
    score = -cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=10).mean()
    # predicted = cross_val_predict(model, X, Y, cv=10)
    p_medio = 100*(abs((Y-prediction).sum())/Y.sum())

    print("\tMean Squared Error: %.4e" % err)
    print("\tMean Error: %.4e" % np.sqrt(err))
    print("\tCross Validation Score (10 fold): %.4e" % score)
    print("\tPorcentaje medio de error: %.4e perc" % p_medio)

    return err, np.sqrt(err), score


def predict_model(model, X: Union[pd.DataFrame, np.ndarray], Y: Union[pd.DataFrame, np.ndarray], y_min: float, y_max: float):
    """Ejecuta un modelo y devuelve la predicción desnormalizada

    :param model: Modelo que usar para hacer la estimación
    :param X: Datos de entrada
    :param Y: Datos de salida
    :param y_min: mínimo antes de haber normalizado
    :param y_max: máximo antes de haber normalizado
    :return: Error cuadrático medio, error medio y putuación de validación cruzada
    """
    model.fit(X, Y)
    prediction = model.predict(X)
    return prediction*(y_max-y_min)+y_min


def main(cables_filename):
    cables = pd.read_csv(cables_filename, sep=",", decimal=".", header=0)

    print(cables.shape)
    # 1. Limpiar el dataset
    cables.dropna(how='any', inplace=True)
    print(cables.shape)
    # 2. Escalar o normalizar las variables
    min_max_scaler = preprocessing.MinMaxScaler()
    c_scaled = min_max_scaler.fit_transform(cables)

    # Máximo y mínimo de la última columna
    c_min = min_max_scaler.data_min_[-1]
    c_max = min_max_scaler.data_max_[-1]

    X_orig = c_scaled[:, :-1]
    Y = c_scaled[:, -1]
    me = 0

    points_x = []
    points_y = ([], [])

    # 3. Detectar las variables irrelevantes o redundantes
    for i in range(1, 6):
        sel = SelectKBest(f_regression, k=i)
        # sel = SelectPercentile(score_func=f_regression, percentile=0.1)
        X = sel.fit_transform(X_orig, Y)

        col_names = cables.columns.values.tolist()[1:]
        print("Con K=%d se mantienen las columnas: %s" % (i, np.extract(sel.get_support(), col_names)))

        # 4. Construir un modelo lineal y otro con random forest
        print("LINEAR REGRESSION")
        mse, me, cv = try_model(LinearRegression(), X, Y)

        print("\nRANDOM FOREST")
        mse2, me2, cv2 = try_model(RandomForestRegressor(), X, Y)

        points_x.append(i)
        points_y[0].append(me)
        points_y[1].append(me2)

    plt.plot(points_x, points_y[0], label='Linear Regression')
    plt.plot(points_x, points_y[1], label='Random Forest')
    plt.legend()
    plt.grid()
    plt.title("Error medio en función de K (K-Best)")
    plt.show()

    # 5. Decidir precisión
    #   x = (y-min)/(max-min) --> y = (max-min)x+min
    me_desnorm = (c_max-c_min)*me+c_min
    print("\nError medio desnormalizado: %.4f metros" % me_desnorm)


if __name__ == '__main__':
    main('cables.csv')
