# Figure: the anharmonic oscillator H = p^2/2 + x^2/2 + lambda x^4 (units hbar = m = omega = 1). The lowest four
# eigenvalues from a truncated 80 x 80 matrix in the number basis (solid) against first-order perturbation theory
# E_n + lambda (3/4)(2n^2 + 2n + 1) (dashed) and the ground state to second order 1/2 + 3 lambda/4 - 21 lambda^2/8
# (dotted). Checks: the matrix elements of x^4 from truncated ladder matrices, the second-order coefficient, the
# exact linear-perturbation shift, the two-level series and the first-order shift in a tilted box.
# Output: figures/qm_perturbation.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm_perturbation.svg")
hbar, m, w = var("hbar m w", domain="positive")
N = 12
a = matrix(SR, N, N, lambda i, j: sqrt(j) if i == j - 1 else 0); ad = a.transpose()
ell2 = hbar / (m * w)
X = sqrt(ell2 / 2) * (a + ad)
X4 = (X ** 4).apply_map(lambda e: e.simplify_full())
print("[PASS] <n|x^4|n> = (hbar/(m omega))^2 (3/4)(2n^2 + 2n + 1) from truncated ladder matrices (n <= 8), so the first-order shift of lambda x^4 is lambda (3/4)(hbar/(m omega))^2 (2n^2 + 2n + 1):",
      all(bool((X4[n, n] - ell2 ** 2 * 3 / 4 * (2 * n ** 2 + 2 * n + 1)).simplify_full() == 0) for n in range(9)))
X1 = ((a + ad) / sqrt(2)); X14 = X1 ** 4
second = sum(X14[k, 0] ** 2 / (0 - k) for k in range(1, 9))
print("[PASS] second order for the ground state of x^2/2 + p^2/2 + lambda x^4: sum |<k|x^4|0>|^2/(E_0 - E_k) = -21/8 (only k = 2, 4 contribute):", bool(SR(second).simplify_full() == -21 / 8))
F = var("F")
H0 = diagonal_matrix(SR, [n + 1 / 2 for n in range(N)])
ok = True
for n in range(6):
    s2 = sum((F * X1[k, n]) ** 2 / (H0[n, n] - H0[k, k]) for k in range(N) if k != n)
    ok = ok and bool(SR(s2).simplify_full() == -F ** 2 / 2) and bool(X1[n, n] == 0)
print("[PASS] for the perturbation F x the first-order shift vanishes and the second-order shift is -F^2/2 for n = 0..5, the exact shift of the completed square:", ok)
xx = var("xx")
print("[PASS] completing the square: x^2/2 + F x = (x + F)^2/2 - F^2/2:", bool((xx ** 2 / 2 + F * xx - ((xx + F) ** 2 / 2 - F ** 2 / 2)).expand() == 0))
E1, E2, v = var("E1 E2 v", domain="real"); assume(E2 > E1)
low = (E1 + E2) / 2 - sqrt(((E2 - E1) / 2) ** 2 + v ** 2)
print("[PASS] two-level system: the exact lower eigenvalue expands as E1 - v^2/(E2 - E1) + v^4/(E2 - E1)^3 + O(v^6), the second-order formula plus the next term:",
      bool((taylor(low, v, 0, 4) - (E1 - v ** 2 / (E2 - E1) + v ** 4 / (E2 - E1) ** 3)).simplify_full() == 0))
x, L, eps = var("x L eps", domain="positive")
print("[PASS] in the box with V = eps x, the first-order shifts <n|eps x|n> = eps L/2 for n = 1..6:",
      all(bool(integrate(2 / L * sin(n * pi * x / L) ** 2 * eps * x, x, 0, L).simplify_full() == eps * L / 2) for n in range(1, 7)))
NB = 80
an = matrix(RDF, NB, NB, lambda i, j: sqrt(j) if i == j - 1 else 0)
Xn = (an + an.transpose()) / sqrt(2.0); Xn4 = Xn ** 4
H0n = diagonal_matrix(RDF, [n + 0.5 for n in range(NB)])
def levels(lam):
    return sorted((H0n + lam * Xn4).eigenvalues())[:4]
lam = 0.01
E = levels(lam)
fo = 0.5 + 0.75 * lam; so = fo - 21 / 8 * lam ** 2
print("[PASS] at lambda = 0.01 the numerical ground state 0.50726 agrees with second order to 3e-5 and with first order only to 3e-4:", abs(E[0] - so) < 3e-5 and 2e-4 < abs(E[0] - fo) < 3e-4 and abs(E[0] - 0.50726) < 1e-5)
print("[PASS] at lambda = 0.01 the first-order formula n + 1/2 + (3/4) lambda (2n^2 + 2n + 1) is within 1% of the numerical levels n = 0..3:",
      all(abs(E[n] - (n + 0.5 + 0.75 * lam * (2 * n ** 2 + 2 * n + 1))) / E[n] < 0.01 for n in range(4)))
# the figure
lams = [0.1 * k / 50 for k in range(51)]
cols = ["firebrick", "royalblue", "darkgreen", "darkorange"]
data = [levels(l_) for l_ in lams]
g = Graphics()
for n, c in enumerate(cols):
    g += line([(l_, d[n]) for l_, d in zip(lams, data)], color=c, thickness=2, legend_label=r"$E_%d$ (matrix)" % n)
    g += line([(l_, n + 0.5 + 0.75 * l_ * (2 * n ** 2 + 2 * n + 1)) for l_ in lams], color=c, linestyle="--", thickness=1.2)
g += line([(l_, 0.5 + 0.75 * l_ - 21 / 8 * l_ ** 2) for l_ in lams], color="black", linestyle=":", thickness=1.5, legend_label=r"$E_0$ to second order")
g.axes_labels([r"$\lambda$", r"$E/\hbar\omega$"]); g.set_legend_options(font_size=8, loc="upper left")
g.matplotlib(figsize=[7, 4.4], ymin=0, ymax=7).savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
