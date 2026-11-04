# Exercise exr-ems-pendulum: two charged balls on threads; equilibrium tan(theta) = F/(m g) with F = q^2/(4 pi eps0 (2 l sin theta)^2).
eps0 = 8.854e-12; m = 1e-3; g = 9.81; l = 0.5; th = 10 * pi / 180
q = 2 * l * sin(th) * sqrt(4 * pi * eps0 * m * g * tan(th))
print("q = %.3e C" % q.n())
F = q^2 / (4 * pi * eps0 * (2 * l * sin(th))^2)
print("check: F/(m g) = %.5f, tan(theta) = %.5f" % (F.n() / (m * g), tan(th).n()))
