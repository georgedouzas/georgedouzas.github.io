# Figure: a rotation by angle phi about the x^3 axis acting on a vector (x^1 component of e_1) and on a spinor (upper
# component of S(phi) psi_0 with S(phi) = exp(-i phi sigma_3 / 2)), for phi in [0, 4 pi]: the vector returns after 2 pi,
# the spinor changes sign at 2 pi and returns after 4 pi. Output: figures/sd-spinor-rotation.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-spinor-rotation.svg")
phi = var("phi")
s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
S = matrix(SR, [[exp(-I * phi / 2), 0], [0, exp(I * phi / 2)]])
Lam = matrix(SR, 3, 3, lambda a, b: (1 / 2 * ((S.conjugate_transpose() * [s1, s2, s3][a] * S) * [s1, s2, s3][b]).trace()).simplify_full())
P = plot(cos(phi), (phi, 0, 4 * pi), color="royalblue", thickness=2, legend_label=r"vector: $\Lambda^1{}_1(\varphi) = \cos\varphi$")
P += plot(cos(phi / 2), (phi, 0, 4 * pi), color="firebrick", thickness=2, legend_label=r"spinor: $\mathrm{Re}\,S(\varphi)_{11} = \cos(\varphi/2)$")
P += plot(-sin(phi / 2), (phi, 0, 4 * pi), color="firebrick", linestyle="--", legend_label=r"spinor: $\mathrm{Im}\,S(\varphi)_{11} = -\sin(\varphi/2)$")
P += point((2 * pi, -1), color="firebrick", size=40) + text(r"$S(2\pi) = -1$", (2 * pi, -1.18), color="firebrick", fontsize=9)
P.axes_labels([r"$\varphi$", ""]); P.set_legend_options(loc="lower left", font_size=8)
P.save(OUT, figsize=[7, 3.8], ymin=-1.35, ymax=1.35, ticks=[pi, 0.5], tick_formatter=[pi, None])
print("[PASS] S(phi)^{-1} sigma_a S(phi) = Lambda_a^b sigma_b with Lambda = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]:", bool(Lam == matrix(SR, [[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])))
print("[PASS] S(2 pi) = -1 and S(4 pi) = 1 while Lambda(2 pi) = 1:", S.subs(phi=2 * pi) == -identity_matrix(2) and S.subs(phi=4 * pi) == identity_matrix(2) and Lam.subs(phi=2 * pi) == identity_matrix(3))
print("[PASS] S(phi) is unitary with determinant one:", bool((S * S.conjugate_transpose()).simplify_full() == identity_matrix(2)) and bool(S.det().simplify_full() == 1))
print("wrote", os.path.relpath(OUT))
