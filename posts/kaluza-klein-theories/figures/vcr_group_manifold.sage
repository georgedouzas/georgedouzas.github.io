# Figure: left-invariant reductions on SU(2) against Killing-vector reductions on S^2. Left: the scalar curvature of
# the left-invariant metric a1^2 s1^2 + a2^2 s2^2 + a3^2 s3^2 on SU(2) along the volume-preserving squashing
# a1 = a2 = t^(-1/2), a3 = t; minus the curvature is the scalar potential of the group-manifold (Scherk-Schwarz)
# reduction, constant over SU(2) and extremal at the round point t = 1. Right: along a meridian of the unit S^2 the
# squared lengths of the three Killing vectors K_a vary from point to point (they enter the non-abelian ansatz of
# Part IV), while the lengths of the left-invariant frame on SU(2) are the constants a_a^2.
# Interactive version: integral curves of the left-invariant field e_3 = d/dpsi on S^3 = SU(2) (Hopf circles),
# stereographically projected to R^3. Output: figures/vcr_group_manifold.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcr_group_manifold.svg")
t, th = var("t th")
Rt = 2 * t - t ** 4 / 2
left = plot(-Rt, (t, 0.35, 1.9), color="firebrick", thickness=2, legend_label=r"$-R(t)$, volume fixed")
left += point((1, -Rt.subs(t=1)), color="black", size=40)
left += text("round $S^3$", (1.0, -Rt.subs(t=1) - 0.35), fontsize=9)
left.axes_labels([r"squashing $t$", r"$V \propto -R$"]); left.set_legend_options(loc="upper center", font_size=8)
right = plot(cos(th) ** 2, (th, 0, pi), color="royalblue", thickness=1.6, legend_label=r"$K_1\cdot K_1$ on $S^2$")
right += plot(1 + 0 * th, (th, 0, pi), color="darkgreen", thickness=1.6, legend_label=r"$K_2\cdot K_2$ on $S^2$")
right += plot(sin(th) ** 2, (th, 0, pi), color="darkorange", thickness=1.6, legend_label=r"$K_3\cdot K_3$ on $S^2$")
right += plot(0.5 + 0 * th, (th, 0, pi), color="black", linestyle="--", legend_label=r"$g(e_a, e_a) = a_a^2$ on $SU(2)$")
right.axes_labels([r"$\theta$ ($\varphi = 0$)", ""]); right.set_legend_options(loc="upper center", ncol=2, font_size=7)
right.ymax(1.45)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10, 4], frame=True, axes=False); fig.savefig(OUT, bbox_inches="tight")

# checks with SageManifolds: SU(2) in Euler angles with the left-invariant forms of Part I (eq-su2-forms)
M = Manifold(3, "SU2"); X = M.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi"); TH, PH, PS = X[:]
s1 = M.one_form(cos(PS), sin(TH) * sin(PS), 0); s2 = M.one_form(-sin(PS), sin(TH) * cos(PS), 0); s3 = M.one_form(0, cos(TH), 1)
sig = [s1, s2, s3]
print("[PASS] Maurer-Cartan equations d sigma_a = -(1/2) eps_abc sigma_b ^ sigma_c for the SU(2) forms in Euler angles:",
      all(sig[c].exterior_derivative() == -sig[a].wedge(sig[b]) for c, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)])))
a1, a2, a3 = var("a1 a2 a3", domain="positive")
g3 = M.metric("g")
coef = [a1 ** 2, a2 ** 2, a3 ** 2]
comps = sum(coef[k] * matrix(SR, 3, 3, lambda i, j: sig[k][i].expr() * sig[k][j].expr()) for k in range(3))
for i in range(3):
    for j in range(i, 3):
        g3[i, j] = comps[i, j].simplify_full()
Rs = g3.ricci_scalar().expr().simplify_full()
Rexp = (2 * a1 ** 2 * a2 ** 2 + 2 * a2 ** 2 * a3 ** 2 + 2 * a3 ** 2 * a1 ** 2 - a1 ** 4 - a2 ** 4 - a3 ** 4) / (2 * a1 ** 2 * a2 ** 2 * a3 ** 2)
print("[PASS] the left-invariant metric sum a_a^2 sigma_a^2 has constant scalar curvature R = (2 a1^2 a2^2 + 2 a2^2 a3^2 + 2 a3^2 a1^2 - a1^4 - a2^4 - a3^4)/(2 a1^2 a2^2 a3^2), R = 6 at a_a = 1/2:",
      bool((Rs - Rexp).simplify_full() == 0) and bool(Rexp.subs(a1=1/2, a2=1/2, a3=1/2) == 6))
E = [M.vector_field() for k in range(3)]
Sm = matrix(SR, 3, 3, lambda a, i: sig[a][i].expr())       # sigma_a = S[a, i] dx^i
Sinv = Sm.inverse().simplify_full()                          # e_a = Sinv[i, a] d_i
for a in range(3):
    E[a][:] = [Sinv[i, a] for i in range(3)]
Ric = g3.ricci()
RicF = matrix(SR, 3, 3, lambda a, b: Ric(E[a], E[b]).expr().simplify_full())
print("[PASS] the frame components Ric(e_a, e_b) of the left-invariant metric are constants (no dependence on theta, phi, psi), diagonal, Ric(e_3, e_3) = (a3^4 - (a1^2 - a2^2)^2)/(2 a1^2 a2^2):",
      all(bool(diff(RicF[a, b], v) == 0) for a in range(3) for b in range(3) for v in [TH, PH, PS])
      and all(bool(RicF[a, b] == 0) for a in range(3) for b in range(3) if a != b)
      and bool((RicF[2, 2] - (a3 ** 4 - (a1 ** 2 - a2 ** 2) ** 2) / (2 * a1 ** 2 * a2 ** 2)).simplify_full() == 0))
tt = var("tt", domain="positive")
Rvol = Rexp.subs(a1=1 / sqrt(tt), a2=1 / sqrt(tt), a3=tt).simplify_full()
print("[PASS] at fixed volume (a1 = a2 = t^(-1/2), a3 = t) R(t) = 2t - t^4/2, stationary and maximal at the round point t = 1 (R = 3/2):",
      bool((Rvol - (2 * tt - tt ** 4 / 2)).simplify_full() == 0) and bool(diff(Rvol, tt).subs(tt=1) == 0) and bool(diff(Rvol, tt, 2).subs(tt=1) < 0))
eps = lambda a, b, c: sign(Permutation([a + 1, b + 1, c + 1])) if len({a, b, c}) == 3 else 0
f_su2 = lambda a, b, c: eps(a, b, c)                                   # [T_a, T_b] = f_ab^c T_c
f_heis = lambda a, b, c: (1 if (a, b, c) == (1, 2, 0) else -1 if (a, b, c) == (2, 1, 0) else 0)   # [t_2, t_y] = m t_1 with m = 1
f_aff = lambda a, b, c: (1 if (a, b, c) == (0, 1, 1) else -1 if (a, b, c) == (1, 0, 1) else 0)     # [t_1, t_2] = t_2
def jacobi(f, n):
    return all(sum(f(a, b, d) * f(d, c, e) + f(b, c, d) * f(d, a, e) + f(c, a, d) * f(d, b, e) for d in range(n)) == 0 for a in range(n) for b in range(n) for c in range(n) for e in range(n))
def trace_ad(f, n):
    return [sum(f(a, b, b) for b in range(n)) for a in range(n)]
print("[PASS] unimodularity f_ab^b = 0 holds for su(2) and the Heisenberg algebra (both satisfy Jacobi), and fails for the two-dimensional algebra [t1, t2] = t2:",
      jacobi(f_su2, 3) and jacobi(f_heis, 3) and trace_ad(f_su2, 3) == [0, 0, 0] and trace_ad(f_heis, 3) == [0, 0, 0] and trace_ad(f_aff, 2) != [0, 0])
# Heisenberg nilmanifold of Hull-Reid-Edwards: g = [[1, -m y, z1], [0, 1, z2], [0, 0, 1]], sigma = g^{-1} dg
N = Manifold(3, "Nil"); XN = N.chart("y z1 z2"); yy, z1, z2 = XN[:]
m = var("m")
gm = matrix(SR, [[1, -m * yy, z1], [0, 1, z2], [0, 0, 1]])
dg = [diff(gm, v) for v in (yy, z1, z2)]
MC = [gm.inverse() * d for d in dg]
sy = N.one_form(1, 0, 0); s1n = N.one_form(*[MC[k][0, 2] for k in range(3)]); s2n = N.one_form(*[MC[k][1, 2] for k in range(3)])
print("[PASS] Heisenberg nilmanifold: g^{-1} dg has entries sigma^y = dy (from -m sigma^y), sigma^1 = dz1 + m y dz2, sigma^2 = dz2, with d sigma^1 = m sigma^y ^ sigma^2 and d sigma^y = d sigma^2 = 0:",
      all(bool(MC[k][0, 1] == -m * [1, 0, 0][k]) for k in range(3)) and s1n.exterior_derivative() == m * sy.wedge(s2n)
      and bool(s1n[1].expr() == 1) and bool(s1n[2].expr() == m * yy) and s2n.exterior_derivative() == 0)
# Killing vectors of the unit S^2 and their inner products
S2 = Manifold(2, "S2"); XS = S2.chart(r"u:(0,pi):\theta v:(0,2*pi):\varphi"); u, v = XS[:]
gs = S2.metric("gs"); gs[0, 0] = 1; gs[1, 1] = sin(u) ** 2
K = [S2.vector_field(-sin(v), -cot(u) * cos(v)), S2.vector_field(cos(v), -cot(u) * sin(v)), S2.vector_field(0, 1)]
nvec = [sin(u) * cos(v), sin(u) * sin(v), cos(u)]
print("[PASS] on the unit S^2 the Killing vectors K_a are Killing, and K_a . K_b = delta_ab - n_a n_b varies over the sphere (only the trace sum_a K_a . K_a = 2 is constant):",
      all(gs.lie_derivative(Ka) == 0 for Ka in K)
      and all(bool((gs(K[a], K[b]).expr() - ((1 if a == b else 0) - nvec[a] * nvec[b])).simplify_full() == 0) for a in range(3) for b in range(3))
      and bool(sum(gs(K[a], K[a]).expr() for a in range(3)).simplify_full() == 2))
print("wrote", os.path.relpath(OUT))

# Interactive version: Hopf circles = integral curves of the left-invariant field e_3 = d/dpsi, stereographically projected
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import threejs_html
s = var("s")
def emb(T, P, S):
    return (cos(T / 2) * cos((P + S) / 2), -cos(T / 2) * sin((P + S) / 2), sin(T / 2) * sin((P - S) / 2), -sin(T / 2) * cos((P - S) / 2))
def proj(T, P):
    x1, x2, x3, x4 = emb(T, P, s)
    return (x1 / (1 - x4), x2 / (1 - x4), x3 / (1 - x4))
scene = Graphics()
palette = [(T, c) for T, c in zip([pi / 6, pi / 2, 5 * pi / 6], ["royalblue", "firebrick", "darkgreen"])]
for T, c in palette:
    for j in range(10):
        P = 2 * pi * j / 10
        scene += parametric_plot3d(proj(T, P), (s, 0, 4 * pi), color=c, thickness=2, plot_points=200)
threejs_html("vcr_group_manifold", scene, frame=False)
print("wrote figures/interactive/vcr_group_manifold.html")
