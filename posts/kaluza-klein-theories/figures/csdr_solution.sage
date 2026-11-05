# Figure: the solution of the constraints [tau, phi_1] = phi_2, [tau, phi_2] = -phi_1 in su(3). Left: the doublet
# phi_+ = p E_13 + q E_23 and the real fields phi_1 = (phi_+ + phi_-)/2, phi_2 = (i/2)(phi_+ - phi_-), drawn as the
# real and imaginary parts of the entries for p = 0.6 + 0.3 i, q = 0.8 - 0.2 i. Right: dimension count of the
# constraint as a linear system on su(3) x su(3) (16 real unknowns, 16 real equations, a 4-dimensional solution space,
# computed by Sage): a complex doublet. Output: figures/csdr_solution.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csdr_solution.svg")
tau = I / 3 * diagonal_matrix(SR, [1, 1, -2])
ad = lambda X, Y: X * Y - Y * X
# general anti-Hermitian traceless 3x3 matrices phi_1, phi_2: 8 real parameters each
def su3(prefix):
    a = [var("%s%d" % (prefix, k), domain="real") for k in range(8)]
    lam = [matrix(SR, [[0, 1, 0], [1, 0, 0], [0, 0, 0]]), matrix(SR, [[0, -I, 0], [I, 0, 0], [0, 0, 0]]), matrix(SR, [[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
           matrix(SR, [[0, 0, 1], [0, 0, 0], [1, 0, 0]]), matrix(SR, [[0, 0, -I], [0, 0, 0], [I, 0, 0]]), matrix(SR, [[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
           matrix(SR, [[0, 0, 0], [0, 0, -I], [0, I, 0]]), matrix(SR, [[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sqrt(3)]
    return sum(I * a[k] * lam[k] for k in range(8)), a
P1, a1 = su3("a"); P2, a2 = su3("b")
eqs = []
for M in (ad(tau, P1) - P2, ad(tau, P2) + P1):
    for e in M.list():
        eqs += [e.real_part(), e.imag_part()]
eqs = [e for e in eqs if e != 0]
A = matrix(SR, [[diff(e, v) for v in a1 + a2] for e in eqs])
rk = A.rank()
print("rank of the constraint system on 16 real unknowns:", rk, "-> solution space of dimension", 16 - rk)
print("[PASS] the constraints leave a 4-dimensional real solution space, one complex SU(2) doublet:", 16 - rk == 4)
sol = A.right_kernel().basis()
print("basis of the solution space (coefficients of lambda_1..lambda_8 for phi_1 | phi_2):"); 
for v in sol: print([x for x in v])
# left panel: the entries of phi_1 and phi_2 in terms of the doublet; right panel: the dimension count
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
E = lambda i, j: matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
ent1 = {(0, 2): r"$p/2$", (1, 2): r"$q/2$", (2, 0): r"$-\bar p/2$", (2, 1): r"$-\bar q/2$"}
ent2 = {(0, 2): r"$ip/2$", (1, 2): r"$iq/2$", (2, 0): r"$i\bar p/2$", (2, 1): r"$i\bar q/2$"}
for x0, ent, title in ((0, ent1, r"$\varphi_1 = (\varphi_+ + \varphi_-)/2$"), (3.8, ent2, r"$\varphi_2 = \frac{i}{2}(\varphi_+ - \varphi_-)$")):
    for i in range(3):
        for j in range(3):
            col = "lightcoral" if (i, j) in ((0, 2), (1, 2)) else ("lightskyblue" if (i, j) in ((2, 0), (2, 1)) else "white")
            ax1.add_patch(plt.Rectangle((x0 + j, -i - 1), 1, 1, facecolor=col, edgecolor="black"))
            ax1.text(x0 + j + 0.5, -i - 0.5, ent.get((i, j), "0"), ha="center", va="center", fontsize=11)
    ax1.text(x0 + 1.5, 0.35, title, ha="center", fontsize=11)
ax1.text(3.4, -3.6, r"$\varphi_+ = pE_{13} + qE_{23}$ (red), $\varphi_- = -\varphi_+^\dagger$ (blue)", ha="center", fontsize=10)
ax1.set_xlim(-0.2, 7.0); ax1.set_ylim(-3.9, 0.8); ax1.set_aspect("equal"); ax1.axis("off")
labels = [(r"unknowns: $(\varphi_1, \varphi_2) \in \mathfrak{su}(3)^2$", 16, "lightsteelblue"), ("independent real equations (rank)", rk, "lightcoral"), ("solutions: Re, Im of $p, q$", 16 - rk, "lightgreen")]
for k, (lab, n, col) in enumerate(labels):
    ax2.barh(-k, n, color=col, edgecolor="black", height=0.6)
    ax2.text(n + 0.3, -k, "%d" % n, va="center", fontsize=11)
ax2.set_yticks([0, -1, -2]); ax2.set_yticklabels([l[0] for l in labels], fontsize=10); ax2.set_xlim(0, 18.5); ax2.set_xlabel("real dimension")
ax2.set_title("the constraints as a linear system (Sage)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# the doublet solves the constraints for every (p, q), and every solution is of this form
p, q = var("p q"); x1, y1, x2, y2 = var("x1 y1 x2 y2", domain="real")
P = (x1 + I * y1) * E(0, 2) + (x2 + I * y2) * E(1, 2); M = -P.conjugate_transpose()
ph1 = (P + M) / 2; ph2 = I * (P - M) / 2
ok_solves = (ad(tau, ph1) - ph2).simplify_full().is_zero() and (ad(tau, ph2) + ph1).simplify_full().is_zero()
print("[PASS] phi_1 = (phi_+ + phi_-)/2, phi_2 = (i/2)(phi_+ - phi_-) with phi_+ = p E_13 + q E_23 solve [tau, phi_1] = phi_2, [tau, phi_2] = -phi_1 for all p, q:", ok_solves)
ok_ah = (ph1.conjugate_transpose() + ph1).simplify_full().is_zero() and (ph2.conjugate_transpose() + ph2).simplify_full().is_zero() and bool(ph1.trace() == 0) and bool(ph2.trace() == 0)
print("[PASS] phi_1 and phi_2 are anti-Hermitian and traceless, elements of su(3):", ok_ah)
gen = [(ph1.subs({x1: 1, y1: 0, x2: 0, y2: 0}), ph2.subs({x1: 1, y1: 0, x2: 0, y2: 0})), (ph1.subs({x1: 0, y1: 1, x2: 0, y2: 0}), ph2.subs({x1: 0, y1: 1, x2: 0, y2: 0})),
       (ph1.subs({x1: 0, y1: 0, x2: 1, y2: 0}), ph2.subs({x1: 0, y1: 0, x2: 1, y2: 0})), (ph1.subs({x1: 0, y1: 0, x2: 0, y2: 1}), ph2.subs({x1: 0, y1: 0, x2: 0, y2: 1}))]
def flat(m1, m2): return vector(SR, sum([[e.real_part(), e.imag_part()] for e in (m1.list() + m2.list())], []))
span_doublet = matrix(SR, [flat(*g) for g in gen])
print("[PASS] the four doublet directions (Re p, Im p, Re q, Im q) are independent, so they span the whole 4-dimensional solution space:", span_doublet.rank() == 4 and 16 - rk == 4)
r2 = (-(ph1 * ph1 + ph2 * ph2).trace()).simplify_full()
print("[PASS] -Tr(phi_1^2 + phi_2^2) = |p|^2 + |q|^2 = r:", bool((r2 - (x1^2 + y1^2 + x2^2 + y2^2)).expand() == 0))
brk = ad(ph1, ph2).simplify_full()
expected = I / 2 * matrix(SR, [[x1^2 + y1^2, (x1 + I * y1) * (x2 - I * y2), 0], [(x2 + I * y2) * (x1 - I * y1), x2^2 + y2^2, 0], [0, 0, -(x1^2 + y1^2 + x2^2 + y2^2)]])
print("[PASS] [phi_1, phi_2] = (i/2) (Phi Phi^dagger (+) -|Phi|^2) in block form:", (brk - expected).simplify_full().is_zero())
# H acts on phi_+ as on a doublet: [X, phi_+] stays in the +i eigenspace for X in the block-diagonal algebra
a11, a12, a21, a22 = var("a11 a12 a21 a22")
X = matrix(SR, [[a11, a12, 0], [a21, a22, 0], [0, 0, -a11 - a22]])
Pp = p * E(0, 2) + q * E(1, 2); C = ad(X, Pp).expand()
print("[PASS] [X, phi_+] for X block diagonal is (U - u) acting on the column (p, q): the doublet transformation of H:",
      all(C[i, j] == 0 for i in range(3) for j in range(3) if (i, j) not in [(0, 2), (1, 2)]) and bool(C[0, 2] == ((a11 + a11 + a22) * p + a12 * q).expand()) and bool(C[1, 2] == (a21 * p + (a22 + a11 + a22) * q).expand()))
