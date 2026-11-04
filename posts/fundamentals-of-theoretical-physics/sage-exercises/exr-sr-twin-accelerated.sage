# Exercise exr-sr-twin-accelerated: a round trip in four phases of 5 years of proper time each at 1 g.
c = 2.99792458e8; g = 9.81; yr = 365.25*86400
cg = c/g/yr; x = 5/cg
print("c/g = %.4f yr, g tau1/c = %.4f" % (cg, x))
print("traveller ages %.1f yr; Earth time %.1f yr; furthest distance %.1f ly" % (20, 4*cg*sinh(x), 2*cg*(cosh(x) - 1)))
