# Exercise exr-clm-hj-gravity: Hamilton-Jacobi for H = p^2/(2m) + m g q, the constant beta = dS/dE and free fall.
q, t, b = var("q t beta", domain="real"); m, g, E = var("m g E", domain="positive")
W = -(2 * m * (E - m * g * q))**(3/2) / (3 * m**2 * g)
S = W - E * t
print("dS/dt + (dS/dq)^2/(2m) + m g q =", (diff(S, t) + diff(S, q)**2 / (2 * m) + m * g * q).simplify_full())
print("beta = dS/dE =", diff(S, E).simplify_full())
qt = E / (m * g) - g * (t + b)**2 / 2
print("q = E/(mg) - g (t + beta)^2/2 satisfies beta = dS/dE for t + beta < 0:", bool(diff(S, E).subs(q=qt).subs(t=-b - 1).simplify_full() == b))
