# Pareto
import numpy
import matplotlib.pyplot as plt
from scipy.stats import uniform

x = numpy.linspace(1, 4, 1000)

# Funcion de densidad
plt.subplot(131)
plt.plot(x, uniform.pdf(x, loc=2, scale=1))  # Probability Distribution Function

# Funcion de distribucion
plt.subplot(132)
plt.plot(x, uniform.cdf(x, loc=2, scale=1))  # Cumulative Distribution Function

# Generador aleatorio
plt.subplot(133)
plt.plot(uniform.rvs(loc=2, scale=1, size=1000))  # Random numbers with pareto distribution
plt.show()
