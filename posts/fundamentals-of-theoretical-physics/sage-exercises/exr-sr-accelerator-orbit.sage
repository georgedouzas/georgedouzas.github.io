# Exercise exr-sr-accelerator-orbit (Blennow-Ohlsson, Problem 1.51): proper acceleration and proper time per turn of
# 7 TeV protons on a circle of circumference 26.7 km.
c = 2.99792458e8; mp = 0.938272; E = 7000.0; C = 26.659e3
g = E/mp; v = sqrt(1 - 1/g^2)*c; r = C/(2*pi.n())
print("gamma = %.1f, r = %.0f m" % (g, r))
print("proper acceleration gamma^2 v^2/r = %.3e m/s^2" % (g^2*v^2/r))
print("laboratory period %.2f microseconds, proper time per turn %.2f ns" % (C/v*1e6, C/v/g*1e9))
