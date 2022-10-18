import numpy
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

Square = numpy.meshgrid(numpy.linspace(-1.1, 1.1, 512), numpy.linspace(-1.1, 1.1, 512), indexing='ij')
X = Square[0]
Y = Square[1]

f = lambda x, y, p: minkowski([x, y], [0.0, 0.0], p) <= 1.0
Ball = lambda p: numpy.vectorize(f)(X, Y, p)
plt.imshow(Ball(3))
plt.axis('off')
plt.show()
