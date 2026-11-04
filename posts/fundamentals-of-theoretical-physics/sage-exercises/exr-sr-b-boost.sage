# Exercise exr-sr-b-boost (Blennow-Ohlsson, Problem 1.130): a pure magnetic field seen from a frame moving along x, and
# the two invariants.
var('Bx By Bz v', domain='real')
g = 1/sqrt(1 - v^2)
Ep = vector([0, -g*v*Bz, g*v*By]); Bp = vector([Bx, g*By, g*Bz])
print("E'.B' =", (Ep*Bp).simplify_full(), "   E'^2 - B'^2 =", (Ep*Ep - Bp*Bp).simplify_full())
