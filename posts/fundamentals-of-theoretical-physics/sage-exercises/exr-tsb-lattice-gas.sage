# Exercise exr-tsb-lattice-gas (Bricmont, Exercise 9.10): the one-dimensional Ising chain H = -J sum s_i s_{i+1} rewritten
# with occupation numbers rho_i = (s_i + 1)/2 is a lattice gas; its grand partition function on a periodic chain of L sites
# is tr T^L with a 2 x 2 transfer matrix; density and pressure in the thermodynamic limit.
b, J, z = var('b J z'); assume(b > 0, J > 0, z > 0)
def w(r1, r2):           # z^((rho + rho')/2) exp(b J s s'), the weight of one bond with half the fugacity on each site
    s1, s2 = 2 * r1 - 1, 2 * r2 - 1
    return z^((r1 + r2) / 2) * exp(b * J * s1 * s2)
Tm = matrix(SR, 2, 2, lambda i, j: w(i, j))
lam = Tm.eigenvalues()
lp = ((1 + z) * exp(b * J) + sqrt((1 - z)^2 * exp(2 * b * J) + 4 * z * exp(-2 * b * J))) / 2
print("transfer matrix:", Tm)
print("largest eigenvalue matches the closed form:", any(bool((l - lp).simplify_full() == 0) for l in lam))
L = 6
import itertools
Xi = sum(prod(w(c[i], c[(i + 1) % L]) for i in range(L)) for c in itertools.product([0, 1], repeat=L))
print("periodic chain, L = 6: Xi = tr T^6:", bool((Xi - (Tm^6).trace()).simplify_full() == 0))
n = (z * diff(log(lp), z)).simplify_full()
print("density at z = 1 (Ising field h = 0):", n.subs(z=1).simplify_full())
for bJ in (0.5, 2.0):
    vals = [n.subs(b=bJ, J=1, z=zz).n(digits=4) for zz in (0.01, 0.1, 1, 10, 100)]
    print("bJ = %.1f: n(z) for z = 0.01, 0.1, 1, 10, 100:" % bJ, vals)
