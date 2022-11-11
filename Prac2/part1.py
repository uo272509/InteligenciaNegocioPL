from sklearn import preprocessing
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from helper import *


def main():
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_informative=3,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               random_state=0,
                               shuffle=False)
    X = pd.DataFrame(X)

    iris_filename = 'datasets-uci-iris.csv'
    letter_filename = 'datasets-uci-letter.csv'

    iris = pd.read_csv(iris_filename, sep=',', decimal='.',
                       header=None, names=['sepal_length', 'sepal_width',
                                           'petal_length', 'petal_width',
                                           'target'])

    letter = pd.read_csv(letter_filename, sep=',', decimal='.',
                         header=0)

    # 1. Eliminación de variables con poca varianza
    drop_var(iris, "iris")
    drop_var(letter, "letter")
    drop_var(X, "X")

    # 2. Eliminación de variables basada en estadísticos univariantes
    drop_stat_univar(iris, "iris")
    drop_stat_univar(letter, "letter")

    print("\tBefore X: %s; Varianza media: %.2f" % (str(X.shape), mean_variance(X)))
    X = scale_df(X)
    print("\tAfter X: %s; Varianza media: %.2f" % (str(X.shape), mean_variance(X)))
    print(X)
    #drop_stat_univar(X, "X")

    # 3. Eliminación recursiva de variables
    estimator = SVC(kernel="linear")
    sel = RFECV(estimator, step=1, cv=5)
    iris_reducido2 = sel.fit(iris.iloc[:, 0:4], iris.iloc[:, 4])
    # print(sel.ranking_)
    # print(sel.support_)


if __name__ == '__main__':
    main()
