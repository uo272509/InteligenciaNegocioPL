import math

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import MDS, locally_linear_embedding, TSNE
from sklearn import datasets
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import euclidean_distances
import seaborn as sns


def PCAa(digits):
    #   1.a) Selecciona las dos variables más dependientes de la clase y haz un gráfico scatter
    kb = SelectKBest(score_func=chi2, k=2)
    X_kb = kb.fit_transform(X=digits.data, y=digits.target)
    print(X_kb.shape)
    plt.scatter(X_kb[:, 0], X_kb[:, 1], c=digits.target, alpha=0.8, edgecolors='none')
    plt.title("Ejercicio 1.a (K-Best)")
    plt.xlabel("Variable más dependiente")
    plt.ylabel("Segunda variable más dependiente")
    plt.show()


def PCAb(digits):
    #   1.b= Aplica a los datos una transformación PCA y haz el gráfico de forma que el eje X sea
    # el valor de la proyección en la componente de mas
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(digits.data)
    print(X_pca.shape)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, alpha=0.8, edgecolors='none')
    plt.title("Ejercicio 1.b (PCA)")
    plt.xlabel("Valor de la proyección en la componente de más varianza")
    plt.ylabel("Valor de la proyección en la segunda componente de más varianza")
    plt.show()


def PCAd(digits, ax, k=32):
    kb = SelectKBest(score_func=chi2, k=k)
    X_kb = kb.fit_transform(X=digits.data, y=digits.target)
    pca = PCA()
    X_pca = pca.fit_transform(X_kb)
    print(X_pca.shape)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, alpha=0.8, edgecolors='none')
    ax.set_title(f"Ejercicio 1.c (K-Best + PCA) con {k} variables")
    # ax.set_xlabel("Valor de la proyección en la componente de más varianza")
    # ax.set_ylabel("Valor de la proyección en la segunda componente de más varianza")


def PLAd_plot(digits, k=None):
    if k is None:
        k = [32, 24, 16, 8, 4, 2]

    fig, ax = plt.subplots(math.floor(math.sqrt(len(k))), math.ceil(math.sqrt(len(k))))
    print(len(ax))
    print(len(ax[0]))
    # TODO: Indexing of subplots
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            PCAd(digits, ax[i, j], k[i+j*len(ax)])

    plt.show()


def LFA_ej2(digits):
    fact_2c = FactorAnalysis(n_components=2)
    X_factor = fact_2c.fit_transform(digits.data)
    plt.scatter(X_factor[:, 0], X_factor[:, 1], c=digits.target, alpha=0.8, edgecolors='none')
    plt.show()


def MDS_ej3(digits):
    mds = MDS(n_components=2, dissimilarity="precomputed")
    similarities = euclidean_distances(digits.data)
    pos = mds.fit(similarities).embedding_
    plt.scatter(pos[:, 0], pos[:, 1], c=digits.target, alpha=0.8,
                edgecolors='none')
    plt.show()


def LLE_ej4(digits):
    data = digits.data
    pos, err = locally_linear_embedding(data, n_neighbors=2, n_components=2)
    plt.scatter(pos[:, 0], pos[:, 1], c=digits.target, alpha=0.8,
                edgecolors='none')
    plt.show()


def t_SNE(digits):
    pca = PCA(n_components=32)
    pca_32 = pca.fit_transform(digits.data)
    tsne = TSNE(n_components=2, random_state=0)
    tsne_res = tsne.fit_transform(pca_32)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], hue=digits.target, palette=sns.hls_palette(10), legend='full')
    plt.show()


def main():
    digits = datasets.load_digits()

    # 1- PCA
    # PCAa(digits)
    # PCAb(digits)
    # PLAd_plot(digits)

    # 2- LFA
    # LFA_ej2(digits)

    # 3- MDS
    # MDS_ej3(digits)

    # 4- LLE
    # LLE_ej4(digits)

    # 5- t-SNE
    t_SNE(digits)


if __name__ == '__main__':
    main()
