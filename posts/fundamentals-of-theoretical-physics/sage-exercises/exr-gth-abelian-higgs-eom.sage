# Exercise exr-gth-abelian-higgs-eom: in the abelian Higgs model with D_mu = d_mu + i e A_mu (1+1 dimensions,
# signature (+, -)) the derivative of the scalar Lagrangian |D phi|^2 - V with respect to A_nu is -e J^nu with the
# gauge-covariant current J^nu = i(phibar D^nu phi - phi (D^nu phi)-bar), which is gauge invariant; Maxwell's
# equations of the model are therefore d_mu F^{mu nu} = e J^nu.
t, x, e, lam, mu = var("t x e lam mu", domain="real"); X = [t, x]; eta = [1, -1]
u, w, al = [function(n)(t, x) for n in ["u", "w", "al"]]
phi = u + I * w; phib = u - I * w
c = list(var("c0 c1", domain="real"))
D = lambda f, A: [diff(f, X[k]) + I * e * A[k] * f for k in range(2)]
Db = lambda f, A: [diff(f, X[k]) - I * e * A[k] * f for k in range(2)]
Lm = lambda f, fb, A: sum(eta[k] * Db(fb, A)[k] * D(f, A)[k] for k in range(2)) - lam / 4 * (fb * f - mu^2)^2
dL = [diff(Lm(phi, phib, c), c[k]) for k in range(2)]                               # d L / d A_k
J = lambda f, fb, A: [I * eta[k] * (fb * D(f, A)[k] - f * Db(fb, A)[k]) for k in range(2)]
print("[PASS] dL/dA_nu = -e J^nu with the covariant current:", all(bool((dL[k] + e * J(phi, phib, c)[k]).expand() == 0) for k in range(2)))
A = [function("A0")(t, x), function("A1")(t, x)]; Ag = [A[k] - diff(al, X[k]) for k in range(2)]
Jg = J(exp(I * e * al) * phi, exp(-I * e * al) * phib, Ag); J0 = J(phi, phib, A)
print("[PASS] the covariant current is gauge invariant:", all(bool((Jg[k] - J0[k]).simplify_full() == 0) for k in range(2)))
print("J^0 on the constant vacuum phi = mu (u = mu, w = 0):", J(SR(mu), SR(mu), c)[0].expand(), " (the 'superconducting' current -2 e mu^2 A^0 sources the field)")
