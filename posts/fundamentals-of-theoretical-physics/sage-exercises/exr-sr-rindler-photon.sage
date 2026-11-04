# Exercise exr-sr-rindler-photon (Blennow-Ohlsson, Problem 1.48 b): the wave vector of a photon in Rindler coordinates.
var('t x a w th', domain='real')
T = arctanh(t/x)/a; X = sqrt(x^2 - t^2)
k = vector([w, w*cos(th), 0, w*sin(th)])            # (t, x, y, z) components
kT = (diff(T, t)*k[0] + diff(T, x)*k[1]).simplify_full()
kX = (diff(X, t)*k[0] + diff(X, x)*k[1]).simplify_full()
print("k^T =", kT, "   k^X =", kX, "   k^Y = 0   k^Z =", k[3])
print("Rindler norm a^2 X^2 (k^T)^2 - (k^X)^2 - (k^Z)^2 =", (a^2*X^2*kT^2 - kX^2 - k[3]^2).simplify_full())
