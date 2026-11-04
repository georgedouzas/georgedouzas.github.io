# Exercise exr-sr-muon-ring (Blennow-Ohlsson, Problem 1.15): mean number of turns of 1 GeV muons in a ring of radius 50 m.
E, m, tau0, R, c = 1000.0, 106.0, 2.2e-6, 50.0, 2.998e8
g = E/m; v = sqrt(1 - 1/g^2)
T = g*tau0; d = v*c*T
print("gamma = %.3f  v/c = %.5f  lab lifetime = %.2f microseconds" % (g, v, T*1e6))
print("distance = %.0f m  circumference = %.1f m  turns = %.1f" % (d, 2*pi.n()*R, d/(2*pi.n()*R)))
