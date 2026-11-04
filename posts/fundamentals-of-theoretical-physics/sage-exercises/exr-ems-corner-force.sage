# Exercise exr-ems-corner-force: a charge q at (a, b) in a grounded right-angled corner; three images.
q, a, b, t = var('q a b t', domain='positive')
P = vector([a, b])
imgs = [(-q, vector([-a, b])), (-q, vector([a, -b])), (q, vector([-a, -b]))]
F = sum(qi * q * (P - Pi) / ((P - Pi).norm()^3 * 4 * pi) for qi, Pi in imgs)
print("F =", F.simplify_full())
Fd = (F.subs(a=t, b=t) * vector([1, 1]) / sqrt(2)).simplify_full()
print("component along the diagonal at (t, t):", Fd, "; equals -q^2 (2 sqrt2 - 1)/(32 pi t^2):", bool((Fd + q^2 * (2 * sqrt(2) - 1) / (32 * pi * t^2)).simplify_full() == 0))
W = integrate(Fd * sqrt(2), t, a, oo)
print("work to bring the charge from infinity to (a, a):", W.simplify_full())
Wim = (q / 2 * sum(qi / (4 * pi * (vector([a, a]) - Pi.subs(b=a)).norm()) for qi, Pi in imgs)).simplify_full()
print("half the charge times the images' potential:", Wim, "; equal:", bool((W - Wim).simplify_full() == 0))
