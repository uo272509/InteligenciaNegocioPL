from typing import Union

from pandas import DataFrame
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectPercentile, f_classif
import numpy as np


def drop_var(dataset, dname):
    dataset_notarget = dataset.drop(dataset.columns[-1], axis=1, inplace=False)

    # Imprimir un resumen del dataset
    print("\nVarianza de %s: %s" % (dname, str(dataset_notarget.shape)))
    v = -1
    for name, values in dataset_notarget.items():
        var = values.var()
        print("\t%s: %.4f" % (name, var))
        if var > v:
            v = var

    # Eliminar columnas con poca varianza
    sel = VarianceThreshold(0.1 * v)
    dataset_reducido = sel.fit_transform(dataset_notarget)

    # Imprimir resumen del dataset
    print("Varianza de %s reducido: %s" % (dname, str(dataset_reducido.shape)))
    for name, values in zip(sel.get_feature_names_out(), dataset_reducido.T):
        print("\t%s: %.4f" % (name, values.var()))

    return dataset_reducido


def drop_stat_univar(dataset, dname):
    dataset_notarget = dataset.drop(dataset.columns[-1], axis=1, inplace=False)

    print("%s:" % dname)
    print("\tBefore: %s; Varianza media: %.2f" % (str(dataset_notarget.shape), mean_variance(dataset_notarget)))

    sel = SelectKBest(chi2, k=2)
    dataset_reducido = sel.fit_transform(dataset.iloc[:, :-1], dataset.iloc[:, -1])
    print("\tSelectKBest chi2: %s; ; Varianza media: %.2f" % (str(dataset_reducido.shape),
                                                              mean_variance(dataset_reducido)))

    sel = SelectPercentile(score_func=chi2, percentile=0.1)
    dataset_reducido2 = sel.fit_transform(dataset.iloc[:, :-1], dataset.iloc[:, -1])
    print("\tSelectPercentile chi2: ; Varianza media: %s; %.2f" % (str(dataset_reducido2.shape),
                                                                   mean_variance(dataset_reducido2)))


def mean_variance(df: Union[DataFrame, np.ndarray]) -> float:
    if type(df) == DataFrame:
        return np.array([x.var() for name, x in df.items()]).mean()
    else:
        return np.array([x.var() for x in df.T]).mean()


def scale_df(df):
    return df.transform(lambda x: x - x.min())
