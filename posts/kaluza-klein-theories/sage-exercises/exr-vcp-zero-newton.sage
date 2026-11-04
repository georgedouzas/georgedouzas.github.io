# Exercise exr-vcp-zero-newton: the Newtonian potential on the Randall-Sundrum brane. The zero mode gives G m1 m2/r; the
# continuum of KK gravitons with masses m > 0, each a Yukawa potential e^{-m r}/r with coupling (G/k)(m/k) (Randall-
# Sundrum, eq. 15), adds int_0^oo dm (G/k)(m/k) e^{-m r}/r = G/(k^2 r^3): V = G m1 m2/r (1 + 1/(k^2 r^2)) (eq. 16).
m, r, k, G = var("m r k G", domain="positive")
corr = integrate(G / k * (m / k) * exp(-m * r) / r, m, 0, oo)
print("KK continuum contribution:", corr, "= G/(k^2 r^3):", bool(corr == G / (k**2 * r**3)))
print("relative correction 1/(k r)^2 for k = 1/(1.6e-35 m) (Planck curvature), r = 1 mm:", (1.6e-35 / 1e-3)**2)
print("for k = 1/(0.1 mm), r = 1 mm the correction is", (1e-4 / 1e-3)**2, ": sub-millimetre tests bound 1/k")
