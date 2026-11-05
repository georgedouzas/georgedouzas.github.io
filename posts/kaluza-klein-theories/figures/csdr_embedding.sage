# Figure: the embedding tau: U(1) -> SU(3) and the centraliser. The 3 x 3 matrix is drawn as blocks: the kernel of
# ad tau (block diagonal, the algebra of H = SU(2) x U(1), 4 real dimensions), the +i eigenspace (upper-right
# column, the doublet phi_+) and the -i eigenspace (lower-left row, its conjugate). Sage checks the eigenvalues of
# ad tau on the matrix units and that the kernel is exactly the block-diagonal algebra.
# Output: figures/csdr_embedding.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csdr_embedding.svg")
E = lambda i, j: matrix(QQbar, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
tau = I / 3 * diagonal_matrix(QQbar, [1, 1, -2])
ad = lambda X, Y: X * Y - Y * X
ev = {}
for i in range(3):
    for j in range(3):
        if i != j:
            M = ad(tau, E(i, j)); ev[(i, j)] = M[i, j]
print("eigenvalues of ad tau on E_ij:", ev)
print("[PASS] ad tau has eigenvalue +i on E_13, E_23, -i on E_31, E_32 and 0 on the block-diagonal units:",
      ev[(0, 2)] == I and ev[(1, 2)] == I and ev[(2, 0)] == -I and ev[(2, 1)] == -I and ev[(0, 1)] == 0 and ev[(1, 0)] == 0)
X = matrix(SR, 3, 3, lambda a, b: var("x%d%d" % (a, b)))
ker = solve([e == 0 for e in (X * tau.change_ring(SR) - tau.change_ring(SR) * X).list()], [X[0, 2], X[1, 2], X[2, 0], X[2, 1]], solution_dict=True)
print("[PASS] the kernel of ad tau is the block-diagonal algebra (x13 = x23 = x31 = x32 = 0):", ker == [{X[0, 2]: 0, X[1, 2]: 0, X[2, 0]: 0, X[2, 1]: 0}])
p = Graphics()
cols = {"H": "lightsteelblue", "+": "lightcoral", "-": "lightskyblue"}
kind = lambda i, j: "+" if (i, j) in [(0, 2), (1, 2)] else ("-" if (i, j) in [(2, 0), (2, 1)] else "H")
labels = {(0, 2): "$\\Phi_1$", (1, 2): "$\\Phi_2$", (2, 0): "$-\\bar\\Phi_1$", (2, 1): "$-\\bar\\Phi_2$"}
for i in range(3):
    for j in range(3):
        p += polygon([(j, -i), (j + 1, -i), (j + 1, -i - 1), (j, -i - 1)], color=cols[kind(i, j)], alpha=0.7, edgecolor="black")
        if (i, j) in labels: p += text(labels[(i, j)], (j + 0.5, -i - 0.5), fontsize=11)
p += text("$\\mathfrak{s}(\\mathfrak{u}(2)\\oplus\\mathfrak{u}(1))$: kernel of $\\mathrm{ad}\\,\\tau$", (1, 0.45), fontsize=9)
p += text("$+i$: $\\varphi_+$", (2.5, 0.45), fontsize=9, color="firebrick")
p += text("$-i$: $\\varphi_- = -\\varphi_+^\\dagger$", (1, -3.45), fontsize=9, color="royalblue")
p += text(r"$\tau = \frac{i}{3}\,\mathrm{diag}(1,1,-2)$", (4.6, -1.5), fontsize=11)
p.axes(False); p.set_aspect_ratio(1); p.xmin(-0.2); p.xmax(6.2); p.ymin(-3.8); p.ymax(0.8)
p.save(OUT, figsize=[7, 4])
print("wrote", os.path.relpath(OUT))

# tau itself, and the group elements exp(alpha tau)
print("[PASS] tau = (i/3) diag(1, 1, -2) is anti-Hermitian and traceless, an element of su(3):", tau.conjugate_transpose() == -tau and tau.trace() == 0)
al = var("al")
expt = diagonal_matrix(SR, [exp(I * al / 3), exp(I * al / 3), exp(-2 * I * al / 3)])
print("[PASS] exp(alpha tau) has determinant 1, is the identity at alpha = 6 pi and equals exp(4 pi i/3) times the identity at alpha = 4 pi (a central element of SU(3)):",
      bool(expt.det().simplify_full() == 1) and expt.subs(al=6 * pi).simplify_full() == identity_matrix(3) and bool((expt.subs(al=4 * pi) - exp(4 * pi * I / 3) * identity_matrix(SR, 3)).simplify_full() == 0))
# the real dimensions: kernel of ad tau inside su(3) and the two eigenspaces
basis_su3 = []
for i in range(3):
    for j in range(i + 1, 3):
        basis_su3 += [E(i, j) - E(j, i), I * (E(i, j) + E(j, i))]
basis_su3 += [I * (E(0, 0) - E(1, 1)), I * (E(1, 1) - E(2, 2))]
def realvec(M): return vector(QQ, sum([[QQ(z.real()), QQ(z.imag())] for z in M.list()], []))
A = matrix(QQ, [realvec(ad(tau, B)) for B in basis_su3])
print("[PASS] inside su(3) (real dimension 8) the kernel of ad tau has real dimension 4 and its image (the two eigenspaces) dimension 4:", A.rank() == 4 and 8 - A.rank() == 4)
