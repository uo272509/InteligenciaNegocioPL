import scipy
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


face = scipy.misc.ascent().astype(float)
faceruido = face + norm.rvs(loc=0, scale=16, size=face.shape)
facemasruido = face + norm.rvs(loc=0, scale=64, size=face.shape)

print(signaltonoise(face))

plt.subplot(131, aspect='equal')
plt.imshow(face)
plt.gray()
plt.subplot(132, aspect='equal')
plt.imshow(faceruido)
plt.gray()
plt.subplot(133, aspect='equal')
plt.imshow(facemasruido)
plt.show()
