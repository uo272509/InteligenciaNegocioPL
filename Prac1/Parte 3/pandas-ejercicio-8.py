import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from fancyimpute import KNN


def test_regression(model, X, Y):
    model.fit(X, Y)
    prediction = model.predict(X)
    err = mean_squared_error(Y, prediction)
    score = -cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=10).mean()
    # predicted = cross_val_predict(LR_model, X, Y, cv=10)
    print("\tMean Squared Error: " + "{:.4e}".format(err))
    print("\tMean Error: $%.2f" % np.sqrt(err))
    print("\tCross Validation Score (10 fold): %.2f" % score)


def test_prediction(model, XC, C):
    model.fit(XC, C)
    # prediction = model.predict(XC)
    score = -cross_val_score(model, XC, C, scoring='neg_mean_squared_error', cv=10).mean()
    print("\tCross Validation Score (10 fold): %.4f" % score)


def ejercicios(df1):
    # 2. Seleccionar con las columnas que son números
    #   (CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard IsActiveMember)
    X = df1.loc[:, ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember"]]

    # 3. Crear una nueva columna llamada EstimatedSalary
    Y = df1.loc[:, "EstimatedSalary"]

    # 4. Hacer tres modelos diferentes (por ej. Regresión Lineal, SVR y Random Forest),
    #   hacer el error cuadrático medio (10 fold) y mirar cuál es mejor

    print("\nLinear Regression:")
    test_regression(LinearRegression(), X, Y)

    print("\nSupport Vector:")
    test_regression(SVR(kernel="poly", C=1e1, epsilon=1), X, Y)

    print("\nRandom Forest:")
    test_regression(RandomForestRegressor(), X, Y)

    # 5. Seleccionar un dataframe 'XC' y cogemos las columnas de antes, pero ahora el salario es uno de los datos y
    #   hacemos un clasificador para predecir con 3 clasificadores distintos la columna 'Exited'
    XC = df1.loc[:, ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
                     "EstimatedSalary"]]

    C = df1.loc[:, "Exited"]

    print("\n\n=============== PREDECIR LA COLUMNA 'Exited' =================\n")

    print("\nNaive Bayes:")
    test_prediction(GaussianNB(), XC, C)

    print("\nQuadratic Discriminant Analysis:")
    test_prediction(QuadraticDiscriminantAnalysis(), XC, C)

    print("\n5 Nearest Neighbors")
    test_prediction(KNeighborsClassifier(5), XC, C)


def main():
    df = pd.read_excel("Churn_Modelling_NANs.xlsx", na_values='NA')

    # Seguir desde aquí
    # 1. Eliminar las filas con valores perdidos (o inputar los valores)
    df1 = df.dropna(how='any', inplace=False)

    ejercicios(df1)

    # 6. Inputar los valores en lugar de eliminarlos y repetir los puntos 4 y 6. Discute las diferencias
    df2 = df.loc[df.loc[:, "EstimatedSalary"].notna(), :]
    df2 = df2.loc[df.loc[:, "Exited"].notna(), :]
    df3 = df2.fillna(inplace=False, value=0)

    # X = df2.loc[:, ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember"]]
    # print(sum(np.isnan(X.values)))
    # X = KNN(k=3, verbose=False).fit_transform(X)
    # print(sum(np.isnan(X)))
    # Y = df2.loc[:, "EstimatedSalary"]

    ejercicios(df3)

    # Imputando la media de las columnas
    cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember"]
    df4 = df2.copy()
    vista = df4.loc[:, cols]
    df4.loc[:, cols] = vista.fillna(value=vista.mean())

    ejercicios(df4)

    df5 = df2.copy()
    vista = df5.loc[:, cols]
    df5.loc[:, cols] = KNN(k=3).fit_transform(vista)

    ejercicios(df5)

    # 7. ¿Cuál crees que es la variable más influyente para el salario? ¿Y para la tasa de abandono?


if __name__ == '__main__':
    main()
