import numpy
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski, euclidean, chebyshev, cdist

Square = numpy.meshgrid(numpy.linspace(-1.1, 1.1, 512), numpy.linspace(-1.1, 1.1, 512), indexing='ij')
X = Square[0]
Y = Square[1]

f = lambda x, y, p: minkowski([x, y], [0.0, 0.0], p) <= 1.0
Ball = lambda p: numpy.vectorize(f)(X, Y, p)

#for i in range(1, 5):
#    plt.subplot(220+i, aspect='equal')
#    plt.imshow(Ball(i))
#    plt.title(f"Minkowski de orden %s" % i)
#    plt.axis('off')
# plt.show()


eu = lambda x, y: euclidean([x, y], [0.0, 0.0]) <= 1.0
BallEu = lambda: numpy.vectorize(eu)(X, Y)

ch = lambda x, y: chebyshev([x, y], [0.0, 0.0]) <= 1.0
BallCh = lambda: numpy.vectorize(ch)(X, Y)

#ma = lambda x, y: cdist([x, y], [0.0, 0.0]) <= 1.0
#BallMa = lambda: numpy.vectorize(ma)(X, Y)

plt.subplot(121, aspect='equal')
plt.imshow(BallEu())
plt.title("Euclídea")
plt.axis('off')
plt.subplot(122, aspect='equal')
plt.imshow(BallCh())
plt.title("Chebyshev")
plt.axis('off')
#plt.subplot(133, aspect='equal')
#plt.imshow(BallMa())
#plt.title("Euclídea de orden %s")
#plt.axis('off')

plt.show()
