# Exercise exr-clm-stationary-second-variation: S(eps) for the oscillator (m = omega = 1) on [0, T] along the paths
# x_cl + eps sin(pi t/T); the coefficient of eps^2 changes sign at T = pi.
t, eps = var("t eps", domain="real"); T = var("T", domain="positive")
xcl = sin(T - t) / sin(T)            # classical path from x(0) = 1 to x(T) = 0
x = xcl + eps * sin(pi * t / T)
S = integrate((diff(x, t)**2 - x**2) / 2, t, 0, T)
c1 = diff(S, eps).subs(eps=0).simplify_full(); c2 = (diff(S, eps, 2) / 2).simplify_full()
print("linear coefficient:", c1)
print("quadratic coefficient:", c2, "== (pi^2 - T^2)/(4T):", bool((c2 - (pi**2 - T**2) / (4 * T)).simplify_full() == 0))
print("sign at T = 1, 3, 4:", [sign(c2.subs(T=v).n()) for v in [1, 3, 4]])
