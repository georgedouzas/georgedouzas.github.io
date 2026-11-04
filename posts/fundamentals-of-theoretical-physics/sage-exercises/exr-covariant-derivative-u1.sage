# Exercise exr-covariant-derivative-u1: the covariant derivative of scalar electrodynamics in 1+1 dimensions
# (signature (+, -)): under phi -> e^{i alpha(x)} phi and A_mu -> A_mu - d_mu alpha, D_mu phi = (d_mu + i A_mu) phi
# transforms as phi does, the Lagrangian (D phi)-bar . D phi = eta^{mu nu} (D_mu phi)-bar D_nu phi is gauge invariant,
# and the current coupled to A is the Noether current (Sage).
t, x = var("t x", domain="real")
u, v, al, A0, A1 = [function(n)(t, x) for n in ["u", "v", "al", "A0", "A1"]]
phi = u + I * v; phib = u - I * v
X = [t, x]; A = [A0, A1]; eta = [1, -1]
D = lambda f, Af: [diff(f, X[mu]) + I * Af[mu] * f for mu in range(2)]
Db = lambda f, Af: [diff(f, X[mu]) - I * Af[mu] * f for mu in range(2)]
phi_g = exp(I * al) * phi; phib_g = exp(-I * al) * phib
A_g = [A[mu] - diff(al, X[mu]) for mu in range(2)]
print("D_mu phi transforms covariantly:", all(bool((D(phi_g, A_g)[mu] - exp(I * al) * D(phi, A)[mu]).simplify_full() == 0) for mu in range(2)))
L = lambda f, fb, Af: sum(eta[mu] * Db(fb, Af)[mu] * D(f, Af)[mu] for mu in range(2))
print("the kinetic Lagrangian (D phi)-bar . D phi is gauge invariant:", bool((L(phi_g, phib_g, A_g) - L(phi, phib, A)).simplify_full() == 0))
a0, a1 = var("a0 a1", domain="real")
Lexp = L(phi, phib, [a0, a1]).expand()
j = [diff(Lexp, av).subs({a0: 0, a1: 0}).simplify_full() for av in (a0, a1)]
jN = [I * eta[mu] * (phib * diff(phi, X[mu]) - phi * diff(phib, X[mu])) for mu in range(2)]     # the Noether current j^mu = i(phibar d^mu phi - phi d^mu phibar)
print("the coefficient of A_mu in the expanded Lagrangian (the current coupled to A):", [jj.expand() for jj in j])
print("it is -j^mu with j^mu the Noether current i(phibar d^mu phi - phi d^mu phibar) = 2(v d^mu u - u d^mu v):", all(bool((j[mu] + jN[mu]).expand() == 0) for mu in range(2)), "; the A^2 'seagull' term is left over")
