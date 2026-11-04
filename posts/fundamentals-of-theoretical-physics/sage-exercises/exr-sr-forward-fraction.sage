# Exercise exr-sr-forward-fraction: the fraction of an isotropic star field seen in the forward hemisphere from a moving
# ship, by Monte Carlo against (1 + v)/2.
import numpy as np
rng = np.random.default_rng(int(8)); v = 0.9
c = rng.uniform(-1, 1, 200000)                          # cos(theta) of the propagation direction in the star frame
cp = (c - v)/(1 - v*c)                                   # aberration to the ship frame
print("fraction seen forward: %.4f  (formula (1 + v)/2 = %.4f)" % (np.mean(cp < 0), (1 + v)/2))
