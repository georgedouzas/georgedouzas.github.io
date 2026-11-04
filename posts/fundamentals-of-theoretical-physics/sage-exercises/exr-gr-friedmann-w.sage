# Exercise exr-gr-friedmann-w: for k = 0 and p = w rho (w > -1), a = t^{2/(3(1 + w))} solves both Friedmann equations with
# rho = 1/(6 pi G (1 + w)^2 t^2), and rho a^{3(1 + w)} is constant.
t, w, G = var('t w G'); assume(t > 0, w > -1, G > 0)
a = t^(2/(3*(1 + w))); H = diff(a, t)/a
rho = 1/(6*pi*G*(1 + w)^2*t^2); p = w*rho
eq1 = (H^2 - 8*pi*G*rho/3).simplify_full()
eq2 = (diff(a, t, 2)/a + 4*pi*G/3*(rho + 3*p)).simplify_full()
print("first Friedmann equation:", bool(eq1 == 0), "; second:", bool(eq2 == 0))
print("rho a^{3(1+w)} =", (rho*a^(3*(1 + w))).simplify_full())
