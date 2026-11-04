# Exercise exr-fan-resolvent-laplace: (lambda - A)^(-1) = int_0^oo e^(-lambda t) e^(tA) dt, for A = sigma_3 and a non-diagonal A.
t, lam = var('t lam'); assume(lam > 3)
for A in (matrix(QQ, [[1, 0], [0, -1]]), matrix(QQ, [[0, 1], [-2, 1]])):
    E = (t * A).exp()
    L = E.apply_map(lambda z: integrate((exp(-lam * t) * z).simplify_full(), t, 0, oo))
    R = (lam - A).inverse()
    print("A =", A.list(), ": Laplace transform equals the resolvent:", bool((L - R).simplify_full() == 0), "; resolvent", R.simplify_full().list())
