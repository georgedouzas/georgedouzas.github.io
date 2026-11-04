# Exercise exr-sr-accelerator-rapidity (Blennow-Ohlsson, Problem 1.72): an electron after a potential difference of
# one megavolt; Lorentz factor, velocity and rapidity, against the Newtonian estimate.
m = 0.511; K = 1.0
g = 1 + K/m
v = sqrt(1 - 1/g^2)
print("gamma = %.4f   v/c = %.4f   rapidity = %.4f" % (g, v, arccosh(g)))
print("Newtonian v/c = sqrt(2K/m) = %.3f" % sqrt(2*K/m))
