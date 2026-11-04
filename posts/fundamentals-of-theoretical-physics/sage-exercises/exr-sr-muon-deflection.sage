# Exercise exr-sr-muon-deflection (Blennow-Ohlsson, Problem 1.128): a 2 GeV muon descending 10 km through a horizontal
# magnetic field of 50 microtesla.
E, m = 2.0, 0.10566; B = 50e-6; h = 1e4
p = sqrt(E^2 - m^2); R = p/(0.299792458*B)          # p in GeV, R in m
print("p = %.4f GeV, R = %.0f m, deflection %.1f m (approximation h^2/2R = %.1f m)" % (p, R, R - sqrt(R^2 - h^2), h^2/(2*R)))
