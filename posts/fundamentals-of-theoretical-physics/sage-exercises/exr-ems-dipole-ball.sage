# Exercise exr-ems-dipole-ball (Pierrus, Question 4.11): magnetic moment of a spinning uniformly charged ball.
r, th, ph, Q, a, w, M = var('r theta phi Q a w M', domain='positive')
rho = 3 * Q / (4 * pi * a^3)
X = r * vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
J = rho * vector([0, 0, w]).cross_product(X)
mz = integrate(integrate(integrate((X.cross_product(J))[2] / 2 * r^2 * sin(th), ph, 0, 2 * pi), th, 0, pi), r, 0, a)
print("m_z =", mz.simplify_full())
L = 2 / 5 * M * a^2 * w
print("m/L =", (mz / L).simplify_full())
