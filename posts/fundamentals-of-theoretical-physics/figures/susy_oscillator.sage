# Figure: the supersymmetric harmonic oscillator, H = omega (a^dag a + b^dag b), one bosonic and one fermionic
# mode. The supercharge Q = sqrt(omega) a^dag b and its adjoint satisfy Q^2 = 0 and {Q, Q^dag} = H on the truncated
# tensor-product space, the simplest realisation of a superalgebra on explicit matrices. The spectrum is
# E = n omega with every level n >= 1 doubly degenerate (one boson-fermion pair) and a unique zero-energy ground
# state: supersymmetry unbroken. Output: figures/susy_oscillator.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "susy_oscillator.svg")
N = 8; omega = 1
a = matrix(SR, N, N, lambda i, j: sqrt(QQ(j)) if j == i + 1 else 0)     # bosonic annihilation (truncated)
b = matrix(SR, [[0, 1], [0, 0]])                                           # fermionic annihilation, {b, b^dag} = 1
one_b = identity_matrix(SR, N); one_f = identity_matrix(SR, 2)
A = a.tensor_product(one_f); Bf = one_b.tensor_product(b)
Ad, Bd = A.transpose(), Bf.transpose()
H = omega * (Ad * A + Bd * Bf)
Q = sqrt(omega) * Ad * Bf; Qd = Q.transpose()
print("[PASS] {b, b^dag} = 1 and b^2 = 0 for the fermionic mode:", (b * b.transpose() + b.transpose() * b == identity_matrix(2)) and (b * b).is_zero())
print("[PASS] Q^2 = 0 and {Q, Q^dag} = H (away from the truncation level):", (Q * Q).is_zero() and
      all((Q * Qd + Qd * Q - H)[i, i] == 0 for i in range(2 * N - 2)))
ev = sorted(H.eigenvalues())
print("[PASS] the spectrum is n omega with the ground state unique and every excited level doubly degenerate:",
      ev[:7] == [0, 1, 1, 2, 2, 3, 3])
p = Graphics()
for n in range(5):
    p += line([(0.2, n), (0.8, n)], color="royalblue", thickness=2)                       # bosonic states |n, 0>
    if n >= 1: p += line([(1.2, n), (1.8, n)], color="firebrick", thickness=2)           # fermionic states |n-1, 1>
    if n >= 1: p += arrow((0.8, n), (1.2, n), color="gray", width=1, arrowsize=3)
    p += text("$E = %d\\,\\omega$" % n, (2.3, n), fontsize=9, horizontal_alignment="left")
p += text("bosonic\n$|n,0\\rangle$", (0.5, -0.8), fontsize=10, color="royalblue")
p += text("fermionic\n$|n-1,1\\rangle$", (1.5, -0.8), fontsize=10, color="firebrick")
p += text("$Q = \\sqrt{\\omega}\\,a^\\dagger b$ pairs the states;\n$|0,0\\rangle$ has no partner", (1.5, 5.0), fontsize=9)
p.axes(False); p.ymin(-1.3); p.ymax(5.6); p.xmin(0); p.xmax(3.2)
p.save(OUT, figsize=[5, 4.5])
print("wrote", os.path.relpath(OUT))
