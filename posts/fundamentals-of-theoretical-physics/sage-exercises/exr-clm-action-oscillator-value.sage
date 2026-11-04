# Exercise exr-clm-action-oscillator-value: the classical action of the oscillator from x1 at t = 0 to x2 at t = T.
t, T, m, w, x1, x2 = var("t T m omega x1 x2", domain="positive")
x = (x1 * sin(w * (T - t)) + x2 * sin(w * t)) / sin(w * T)
print("x'' + omega^2 x = 0:", bool((diff(x, t, 2) + w**2 * x).simplify_full() == 0), "; x(0) = x1:", bool(x.subs(t=0) == x1), "; x(T) = x2:", bool(x.subs(t=T).simplify_full() == x2))
S = integrate(m / 2 * (diff(x, t)**2 - w**2 * x**2), t, 0, T)
target = m * w / (2 * sin(w * T)) * ((x1**2 + x2**2) * cos(w * T) - 2 * x1 * x2)
print("S equals m omega/(2 sin omega T) [(x1^2 + x2^2) cos omega T - 2 x1 x2]:", bool((S - target).simplify_full().trig_reduce().simplify_full() == 0))
