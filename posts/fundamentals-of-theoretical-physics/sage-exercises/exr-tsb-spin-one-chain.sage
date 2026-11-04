# Exercise exr-tsb-spin-one-chain (Bricmont, Exercise 9.9): the one-dimensional Ising chain with spins s_i in {-1, 0, 1},
# H = -J sum s_i s_{i+1}; 3 x 3 transfer matrix, its eigenvalues, the free energy per spin -T log lambda_max, and the
# absence of a phase transition, compared with the mean-field critical temperature 2zJ/3 = 4J/3.
b, J = var('b J'); assume(b > 0, J > 0)
S = [-1, 0, 1]
Tm = matrix(SR, 3, 3, lambda i, j: exp(b * J * S[i] * S[j]))
lmax = (1 + 2 * cosh(b * J) + sqrt((2 * cosh(b * J) - 1)^2 + 8)) / 2
lam = [lmax, (1 + 2 * cosh(b * J) - sqrt((2 * cosh(b * J) - 1)^2 + 8)) / 2, 2 * sinh(b * J)]
print("eigenvalues of T agree with the closed forms at bJ = 0.3, 1, 2.5:",
      all(sorted(Tm.subs(b=v, J=1).change_ring(RDF).eigenvalues()) == sorted(Tm.subs(b=v, J=1).change_ring(RDF).eigenvalues()) and
          all(abs(x - y) < 1e-9 for x, y in zip(sorted(Tm.subs(b=v, J=1).change_ring(RDF).eigenvalues()), sorted([l.subs(b=v, J=1).n() for l in lam])))
          for v in (0.3, 1, 2.5)))
import itertools
L = 5
Zp = sum(exp(b * J * sum(c[i] * c[(i + 1) % L] for i in range(L))) for c in itertools.product(S, repeat=L))
print("periodic chain, L = 5: Z = sum of lambda^5 at bJ = 0.7:", bool(abs((Zp - sum(l^5 for l in lam)).subs(b=0.7, J=1).n()) < 1e-8))
fT = lambda Tv: (-Tv * log(lmax.subs(b=1 / Tv, J=1))).n()
print("free energy per spin f(T)/J at T/J = 0.5, 1, 4/3, 2:", [fT(t).n(digits=6) for t in (0.5, 1, 4 / 3, 2)])
print("lambda_max - other eigenvalues > 0 for all T > 0 (gap at T/J = 0.2, 1, 5):",
      [(lmax - max(lam[1], lam[2])).subs(b=1 / t, J=1).n(digits=5) for t in (0.2, 1, 5)])
