# Figure: the non-abelian gauge transformation as a spacetime-dependent rotation of the sphere fibres. Left: the
# same point of the sphere (black) carried by the rotations exp(eps^a(x) K_a) with eps(x) varying along x, at four
# values of x; a field on the sphere is relabelled by a different rotation at each x. Right: the induced change of
# the gauge field, delta A^c = d_mu eps^c + eps_abc A^a eps^b, for a parameter eps^3(x) alone acting on a constant
# A^1: the component A^2 is generated (the commutator term), as checked symbolically from the Lie derivative of the
# ansatz metric. Output: figures/nonabelian_gauge_rotation.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nonabelian_gauge_rotation.svg")
th, ph = var("th ph")
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
h = diagonal_matrix([1, sin(th)^2])
def lc(a, b, c): return Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
A = [[var("A%d%d" % (a, mu)) for mu in range(4)] for a in range(3)]; eps = [var("e%d" % a) for a in range(3)]; deps = [[var("de%d%d" % (a, mu)) for mu in range(4)] for a in range(3)]
Klow = [h * K[a] for a in range(3)]
ok = True
for mu in range(4):
    ghat = [sum(Klow[a][m] * A[a][mu] for a in range(3)) for m in range(2)]
    xi = [sum(eps[a] * K[a][m] for a in range(3)) for m in range(2)]
    dxi = [sum(deps[a][mu] * K[a][m] for a in range(3)) for m in range(2)]
    delta = [sum(xi[n] * diff(ghat[m], [th, ph][n]) for n in range(2)) + sum(dxi[n] * h[n, m] for n in range(2)) + sum(diff(xi[n], [th, ph][m]) * ghat[n] for n in range(2)) for m in range(2)]
    dA = [deps[c][mu] + sum(lc(a, b, c) * A[a][mu] * eps[b] for a in range(3) for b in range(3)) for c in range(3)]
    cand = [sum(Klow[c][m] * dA[c] for c in range(3)) for m in range(2)]
    ok &= all(bool((delta[m] - cand[m]).simplify_full() == 0) for m in range(2))
print("[PASS] L_xi g_{mu m} = K_{m c} (d_mu eps^c + eps_abc A^a_mu eps^b) for xi = eps^a(x) K_a, all mu:", ok)
# left panel: a point carried by rotations about the z axis (eps^3(x)) and the x axis (eps^1(x))
xs = var("xs")
eps3 = lambda xv: 1.2 * sin(xv); eps1 = lambda xv: 0.5 * xv
def rot(axis, ang):
    c, s = cos(ang), sin(ang)
    return {0: matrix([[1, 0, 0], [0, c, -s], [0, s, c]]), 2: matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]
P0 = vector([0.0, 0.0, 1.0])
left = Graphics()
for i, xv in enumerate([-1.5, -0.5, 0.5, 1.5]):
    cx = 2.4 * i
    left += circle((cx, 0), 1, color="steelblue")
    P = rot(0, eps1(xv)) * rot(2, eps3(xv)) * vector([0.6, 0.0, 0.8])
    left += point((cx + P[0], P[2]), size=40, color="black")
    left += point((cx + 0.6, 0.8), size=25, color="gray")
    left += line([(cx + 0.6, 0.8), (cx + P[0], P[2])], color="firebrick", linestyle="--")
    left += text("$x = %.1f$" % xv, (cx, -1.3), fontsize=9)
left += text(r"the gray point relabelled by $e^{\epsilon^a(x) K_a}$", (3.6, 1.45), fontsize=9)
left.axes(False); left.set_aspect_ratio(1)
right = plot(0.6 + 0 * xs, (xs, -2, 2), color="gray", thickness=2, legend_label="$A^1$ (constant)")
right += plot(diff(eps3(xs), xs), (xs, -2, 2), color="royalblue", thickness=2, legend_label=r"$\delta A^3 = \partial_x\epsilon^3$")
right += plot(0.6 * eps3(xs), (xs, -2, 2), color="firebrick", thickness=2, legend_label=r"$\delta A^2 = \epsilon_{132}A^1\epsilon^3 = -A^1\epsilon^3$")
right.axes_labels(["$x$", ""]); right.set_legend_options(loc="upper left", font_size=7); right.ymax(2.2); right.ymin(-0.9)
fig = graphics_array([left, right]).matplotlib(figsize=[10, 3.6]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
