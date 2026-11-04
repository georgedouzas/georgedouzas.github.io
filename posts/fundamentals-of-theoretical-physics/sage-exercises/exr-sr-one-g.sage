# Exercise exr-sr-one-g: proper time, coordinate time and distance to reach 0.99 c at a proper acceleration of 9.81 m/s^2.
c = 2.99792458e8; g = 9.81; yr = 365.25*86400
cg = c/g/yr
eta = arctanh(0.99)
print("c/g = %.4f yr; tau = %.3f yr, t = %.3f yr, x = %.3f ly" % (cg, cg*eta, cg*sinh(eta), cg*(cosh(eta) - 1)))
