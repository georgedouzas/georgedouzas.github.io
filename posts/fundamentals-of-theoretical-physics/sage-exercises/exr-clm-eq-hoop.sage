# Exercise exr-clm-eq-hoop: equilibria and small-oscillation frequencies of a bead on a hoop rotating at angular
# velocity Omega, with effective potential V = -m g R cos(theta) - m R^2 Omega^2 sin^2(theta)/2 and kinetic matrix m R^2.
th = var("theta", domain="real"); m, g, R, W = var("m g R Omega", domain="positive")
V = -m * g * R * cos(th) - m * R**2 * W**2 * sin(th)**2 / 2
print("V' =", diff(V, th).factor())
print("V''(0) =", diff(V, th, 2).subs(theta=0).factor(), " -> the bottom is stable iff Omega^2 < g/R")
c0 = g / (R * W**2)
Vpp = diff(V, th, 2).subs(theta=arccos(c0)).simplify_full()
print("V''(theta0) at cos(theta0) = g/(R Omega^2):", Vpp, "== m R^2 Omega^2 sin^2(theta0):", bool((Vpp - m * R**2 * W**2 * (1 - c0**2)).simplify_full() == 0))
print("omega^2 = V''/(m R^2) =", (Vpp / (m * R**2)).simplify_full())
