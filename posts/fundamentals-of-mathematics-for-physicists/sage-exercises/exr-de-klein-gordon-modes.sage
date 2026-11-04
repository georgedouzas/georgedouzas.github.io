# Exercise exr-de-klein-gordon-modes: (a) the Yukawa solution of the Helmholtz equation for omega < m; (b) the box
# frequencies sqrt(n1^2 + n2^2 + n3^2 + m^2) and the lowest sqrt(3 + m^2); (c) J_0(m sqrt(t^2 - x^2)) solves
# (d_t^2 - d_x^2 + m^2) u = 0 inside the light cone and equals 1 on it.
t, x, y, z, r, m, w = var("t x y z r m omega")
assume(m > 0, r > 0, m > w, w > 0)
mu = sqrt(m ** 2 - w ** 2)
v = exp(-mu * r) / r
helm = -(diff(v, r, 2) + 2 * diff(v, r) / r) - (w ** 2 - m ** 2) * v
print("(a) -Laplacian v = (omega^2 - m^2) v for v = e^(-mu r)/r, mu = sqrt(m^2 - omega^2):", bool(helm.simplify_full() == 0))
n1, n2, n3 = var("n1 n2 n3")
mode = exp(-I * w * t) * sin(n1 * x) * sin(n2 * y) * sin(n3 * z)
kg = (diff(mode, t, 2) - diff(mode, x, 2) - diff(mode, y, 2) - diff(mode, z, 2) + m ** 2 * mode) / mode
print("(b) (Box + m^2) mode / mode =", kg.simplify_full(), "-> omega^2 = n1^2 + n2^2 + n3^2 + m^2")
print("    lowest frequency (n = (1, 1, 1)):", sqrt(3 + m ** 2), "| the mode vanishes on the faces x = 0, pi for integer n1:", bool(mode.subs(x=0) == 0) and bool(mode.subs(x=pi, n1=2).simplify_full() == 0))
s = sqrt(t ** 2 - x ** 2)
u = bessel_J(0, m * s)
res = (diff(u, t, 2) - diff(u, x, 2) + m ** 2 * u).simplify_full()
print("(c) Sage expresses the residual through J_0, J_1, J_2:", res)
res2 = res.subs(bessel_J(2, m * s) == 2 * bessel_J(1, m * s) / (m * s) - bessel_J(0, m * s)).simplify_full()   # the recurrence J_0 + J_2 = 2 J_1/(ms)
print("    after the recurrence J_0 + J_2 = (2/ms) J_1 the residual is", res2, "-> zero inside the cone:", bool(res2 == 0))
print("    numerical check at (t, x, m) = (3, 1, 1.3):", abs(res.subs(t=3, x=1, m=1.3).n()) < 1e-12)
print("    value on the light cone (s = 0): J_0(0) =", bessel_J(0, 0), "| for m = 0 the factor is identically 1:", bool(u.subs(m=0) == 1))
