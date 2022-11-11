import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)
X = pd.DataFrame(X)

# Clasificador basado en arboles de decision
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
# X, y son las variables de entrada y de salida del dataset
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in
              forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Variables ordenadas por importancia
print("Variables ordenadas:")
for f in range(X.shape[1]):
    print("%d. variable %d (%f)" % (f + 1, indices[f],
                                    importances[indices[f]]))
# Grafico con las importancias de las variables
plt.figure()
plt.title("Importancia de las variables")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
