# Exercise exr-sr-thomas-bohr: Thomas precession of the electron on the first Bohr orbit, compared with the orbital
# angular velocity.
c = 2.99792458e8; alpha = 1/137.036; r = 0.529177e-10
v = alpha*c; a = v^2/r; g = 1/sqrt(1 - alpha^2)
wT = g^2/(g + 1)*a*v/c^2; w = v/r
print("v = %.3e m/s, a = %.3e m/s^2" % (v, a))
print("omega_T = %.3e rad/s, orbital omega = %.3e rad/s, ratio = %.3e, alpha^2/2 = %.3e" % (wT, w, wT/w, alpha^2/2))
