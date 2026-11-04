# Exercise exr-ems-cyclotron-numbers: proton in 1.5 T; cyclotron frequency and orbit radius at 10 MeV (SI units).
e, mp, c = 1.602e-19, 1.673e-27, 2.998e8
B = 1.5; K = 10e6 * e
f = e * B / (2 * pi.n() * mp); v = sqrt(2 * K / mp); r = mp * v / (e * B)
print("f = %.2f MHz, v = %.3e m/s (v/c = %.3f), r = %.3f m" % (f / 1e6, v, v / c, r))
print("relativistic gamma - 1 = K/(m c^2) = %.4f" % (K / (mp * c^2)))
