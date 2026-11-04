# Exercise exr-ren-x-uehling-potential: the Uehling correction int_1^oo e^(-2 m r u) (1 + 1/(2u^2)) sqrt(u^2 - 1)/u^2 du
# and its short-distance form log(1/(m r)) - gamma_E - 5/6.
from scipy.integrate import quad
import numpy as np
U = lambda mr: quad(lambda u: np.exp(-2 * mr * u) * (1 + 1 / (2 * u * u)) * np.sqrt(u * u - 1) / u**2, 1, np.inf, limit=500)[0]
for mr in [0.001, 0.01, 0.1, 1.0]:
    print("m r = %-6s correction %.4f   asymptotic %.4f" % (mr, U(mr), np.log(1 / mr) - float(euler_gamma) - 5 / 6))
