# Figure: a free scalar field on a circle of length L = 2 pi (wave numbers k_n = n, mass m = 1). Left: the mode
# functions Re(e^{i k_n x})/sqrt(2 omega_n L) of the first modes, with the vacuum amplitude 1/sqrt(2 omega_n L) of
# each mode; right: the mode sum (1/L) sum_{|n| <= N} e^{i k_n s} of the canonical commutator [phi(x), pi(y)] / i,
# the Dirichlet kernel, for N = 2, 5, 10, concentrating at s = 0 as the truncation is lifted.
# The checks use truncated oscillator matrices for the modes. Output: figures/free_field_modes.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "free_field_modes.svg")
L = 2 * pi; m = 1
x, y, s = var("x y s", domain="real")
k = lambda n: 2 * pi * n / L
om = lambda n: sqrt(k(n) ** 2 + m ** 2)
left = Graphics()
for n in range(4):
    amp = 1 / sqrt(2 * om(n) * L)
    left += plot(amp * cos(k(n) * x), (x, 0, L), color=["black", "royalblue", "darkgreen", "firebrick"][n], thickness=1.4,
                 legend_label=r"$n=%d$, $\omega_n=%s$" % (n, latex(om(n))))
left.axes_labels([r"$x$", r"$\mathrm{Re}\,e^{ik_nx}/\sqrt{2\omega_nL}$"])
left.set_legend_options(loc="lower left", font_size=7)
D = lambda N: sum(exp(I * k(n) * s) for n in range(-N, N + 1)) / L
Dclosed = lambda N: sin((2 * N + 1) * pi * s / L) / (L * sin(pi * s / L))
right = Graphics()
for N, c in zip([2, 5, 10], ["royalblue", "darkorange", "firebrick"]):
    right += plot(Dclosed(N), (s, -pi, pi), color=c, thickness=1.2, legend_label=r"$N=%d$" % N, plot_points=400)
right.axes_labels([r"$s=x-y$", r"$[\phi(x),\pi(y)]/i$"])
right.set_legend_options(loc="upper left", font_size=8)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10, 3.8]); fig.subplots_adjust(wspace=0.35); fig.savefig(OUT, bbox_inches="tight")

# 1. truncated oscillators of two modes: a_1 = a (x) 1, a_2 = 1 (x) a
M = 4
a = matrix(QQbar, M, M, lambda i, j: sqrt(j) if j == i + 1 else 0)
one = identity_matrix(QQbar, M)
a1 = a.tensor_product(one); a2 = one.tensor_product(a)
c11 = a1 * a1.conjugate_transpose() - a1.conjugate_transpose() * a1
c12 = a1 * a2.conjugate_transpose() - a2.conjugate_transpose() * a1
off = [i for i in range(M * M) if i // M < M - 1]
print("[PASS] truncated oscillators of different modes commute and [a_n, a_n^dagger] = 1 off the truncation:",
      c12 == 0 and all(c11[i, i] == 1 for i in off) and all(c11[i, j] == 0 for i in range(M * M) for j in range(M * M) if i != j))
# 2. orthonormal mode functions
ok = all(bool(integrate(exp(I * k(p) * x) * exp(-I * k(q) * x), x, 0, L) / L == (1 if p == q else 0)) for p in range(-3, 4) for q in range(-3, 4))
print("[PASS] the mode functions e^{i k_n x}/sqrt(L) are orthonormal on the circle:", ok)
# 3. the commutator mode sum from truncated oscillators acting on the vacuum (N = 2: five modes, two levels each)
N = 2
b = matrix(SR, [[0, 1], [0, 0]]); e2 = identity_matrix(SR, 2)
def op(j):
    mats = [b if i == j else e2 for i in range(2 * N + 1)]
    out = mats[0]
    for X in mats[1:]:
        out = out.tensor_product(X)
    return out
ops = [op(j) for j in range(2 * N + 1)]
modes = list(range(-N, N + 1))
phi = sum((ops[j] * exp(I * k(n) * x) + ops[j].transpose() * exp(-I * k(n) * x)) / sqrt(2 * om(n) * L) for j, n in enumerate(modes))
piy = sum(-I * om(n) * (ops[j] * exp(I * k(n) * y) - ops[j].transpose() * exp(-I * k(n) * y)) / sqrt(2 * om(n) * L) for j, n in enumerate(modes))
vac = vector(SR, [1] + [0] * (2 ** (2 * N + 1) - 1))
lhs = (phi * piy - piy * phi) * vac
rhs = I * D(N).subs(s=x - y) * vac
print("[PASS] the mode sum of [phi(x), pi(y)] on the vacuum is i times the Dirichlet kernel (1/L) sum_{|n|<=N} e^{i k_n (x-y)}:",
      all(bool((lhs[i] - rhs[i]).expand().simplify_full() == 0) for i in range(len(vac))))
# telescoping: (sum_{|n|<=N} e^{i n s}) (e^{i s/2} - e^{-i s/2}) = e^{i (N+1/2) s} - e^{-i (N+1/2) s}, i.e. the closed form
print("[PASS] (1/L) sum_{|n|<=N} e^{i k_n s} = sin((2N+1) pi s/L)/(L sin(pi s/L)) for N = 1..5:",
      all(bool((L * D(n) * (exp(I * s / 2) - exp(-I * s / 2))).expand() == (exp(I * (n + QQ(1) / 2) * s) - exp(-I * (n + QQ(1) / 2) * s)).expand()) for n in range(1, 6))
      and bool(Dclosed(1) == sin(3 * s / 2) / (2 * pi * sin(s / 2))))
print("[PASS] the Dirichlet kernel integrates to 1 over the circle for N = 1..5:", all(bool(integrate(D(n), s, 0, L) == 1) for n in range(1, 6)))
# 5. Fock energies: three modes, three levels each, H = sum omega_n (a^dagger a + 1/2)
M = 3
a = matrix(SR, M, M, lambda i, j: sqrt(j) if j == i + 1 else 0); one = identity_matrix(SR, M)
A = [a.tensor_product(one).tensor_product(one), one.tensor_product(a).tensor_product(one), one.tensor_product(one).tensor_product(a)]
w = [om(0), om(1), om(2)]
H = sum(w[j] * (A[j].transpose() * A[j] + identity_matrix(SR, M ** 3) / 2) for j in range(3))
E0 = sum(w) / 2
okE = all(bool(H[i, i] - (E0 + sum(nj * wj for nj, wj in zip([i // 9, (i // 3) % 3, i % 3], w))) == 0) for i in range(M ** 3)) and H.is_diagonal()
print("[PASS] on the Fock states |n_0, n_1, n_2> the Hamiltonian sum omega_n (a_n^dagger a_n + 1/2) is diagonal with energy E_0 + sum n_k omega_k:", okE)
print("wrote", os.path.relpath(OUT))
