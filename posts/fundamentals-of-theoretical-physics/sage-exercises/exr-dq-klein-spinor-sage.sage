# Exercise exr-dq-klein-spinor-sage: the three plane waves of the Klein step solve (sigma_1 p + sigma_3 m + V - E) psi = 0, p = -i d/dx,
# on their sides, given k^2 = E^2 - m^2 and q^2 = (E - V)^2 - m^2.
x, E, V, m, k, q = var("x E V m k q")
s1 = matrix(SR, [[0, 1], [1, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
op = lambda psi, pot: (-I * s1 * diff(psi, x) + (s3 * m + (pot - E) * identity_matrix(SR, 2)) * psi)
waves = [("incident", vector(SR, [1, k / (E + m)]) * exp(I * k * x), 0, k^2 - (E^2 - m^2)),
         ("reflected", vector(SR, [1, -k / (E + m)]) * exp(-I * k * x), 0, k^2 - (E^2 - m^2)),
         ("transmitted", vector(SR, [1, q / (E - V + m)]) * exp(I * q * x), V, q^2 - ((E - V)^2 - m^2))]
for name, psi, pot, rel in waves:
    res = [(c * exp(-I * k * x) if name == "incident" else c * exp(I * k * x) if name == "reflected" else c * exp(-I * q * x)).simplify_full() for c in op(psi, pot)]
    # each component is a multiple of the dispersion relation
    sym = k if name != "transmitted" else q
    red = [c.subs({sym^2: (E^2 - m^2) if name != "transmitted" else ((E - V)^2 - m^2)}).simplify_full() for c in [r.numerator() for r in res]]
    print(name, ": residual numerators", res, "; vanish on the mass shell:", all(bool(SR(r).subs(solve(rel == 0, sym)[0]).simplify_full() == 0) for r in res))
