# Figure: Gaussian integrals and the thermal circle. Left: the three Wick pairings of four points x1..x4, which give
# <x1 x2 x3 x4> = G12 G34 + G13 G24 + G14 G23 for a Gaussian weight. Right: the Euclidean propagator of one mode of
# frequency omega = 1 on a thermal circle of circumference beta, G_beta(tau) = cosh(omega(beta/2 - tau))/(2 omega
# sinh(omega beta/2)) extended periodically, for beta = 1, 2, 4, and the zero-temperature limit e^{-omega|tau|}/(2 omega).
# Output: figures/thermal_propagator.svg.
import os, itertools
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thermal_propagator.svg")
left = Graphics()
pairings = [[(1, 2), (3, 4)], [(1, 3), (2, 4)], [(1, 4), (2, 3)]]
cols = ["royalblue", "darkgreen", "firebrick"]
for r, (P, c) in enumerate(zip(pairings, cols)):
    y0 = -2.2 * r
    for (i, j) in P:
        mid = (i + j) / 2; rad = (j - i) / 2
        left += arc((mid, y0), rad, sector=(0, pi), color=c, thickness=1.6)
    left += point([(i, y0) for i in range(1, 5)], color="black", size=30)
    for i in range(1, 5):
        left += text(r"$x_%d$" % i, (i, y0 - 0.35), fontsize=9, color="black")
    left += text(r"$G_{%d%d}G_{%d%d}$" % (P[0][0], P[0][1], P[1][0], P[1][1]), (5.3, y0 + 0.4), fontsize=10, color=c)
left.axes(False); left.set_aspect_ratio(1)
tau = var("tau", domain="real")
right = Graphics()
Gb = lambda b: cosh(b / 2 - tau) / (2 * sinh(b / 2))
for b, c in zip([1, 2, 4], ["firebrick", "darkorange", "royalblue"]):
    for kk in range(0, ceil(6 / b)):
        right += plot(Gb(b).subs(tau=tau - kk * b), (tau, kk * b, min((kk + 1) * b, 6)), color=c, thickness=1.4,
                      legend_label=(r"$\beta=%d$" % b if kk == 0 else None))
right += plot(exp(-tau) / 2, (tau, 0, 6), color="black", linestyle="--", legend_label=r"$\beta=\infty$: $e^{-\omega\tau}/2\omega$")
right.axes_labels([r"$\tau$", r"$G_\beta(\tau)$"]); right.set_axes_range(ymin=0, ymax=1.5); right.set_legend_options(loc="upper right", font_size=8)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10, 3.8]); fig.subplots_adjust(wspace=0.25); fig.savefig(OUT, bbox_inches="tight")

# 1. the finite-dimensional Gaussian integral with a source
x1, x2, J1, J2 = var("x1 x2 J1 J2")
A = matrix(QQ, [[2, 1], [1, 3]]); X = vector([x1, x2]); Jv = vector([J1, J2])
integrand = exp(-X * A * X / 2 + Jv * X)
Z = integrate(integrate(integrand, x1, -oo, oo), x2, -oo, oo)
Zexp = 2 * pi / sqrt(A.det()) * exp(Jv * A.inverse() * Jv / 2)
print("[PASS] int d^2x exp(-x.A.x/2 + J.x) = (2 pi)/sqrt(det A) exp(J.A^{-1}.J/2) for A = [[2,1],[1,3]]:", bool((Z / Zexp).simplify_full() == 1))
# 2. four-point functions from the generating function exp(J.G.J/2), three variables, symbolic symmetric G
n = 3
Gs = matrix(SR, n, n, lambda i, j: var("G%d%d" % (min(i, j), max(i, j))))
Js = vector([var("j%d" % i) for i in range(n)])
W = exp(Js * Gs * Js / 2)
def corr(idx):
    e = W
    for i in idx:
        e = diff(e, Js[i])
    return e.subs({Js[i]: 0 for i in range(n)})
ok4 = all(bool(corr([i, j, k, l]) == Gs[i, j] * Gs[k, l] + Gs[i, k] * Gs[j, l] + Gs[i, l] * Gs[j, k])
          for i, j, k, l in itertools.product(range(n), repeat=4))
ok3 = all(bool(corr([i, j, k]) == 0) for i, j, k in itertools.product(range(n), repeat=3))
print("[PASS] derivatives of exp(J.G.J/2): <x_i x_j x_k x_l> = G_ij G_kl + G_ik G_jl + G_il G_jk and odd moments vanish:", ok4 and ok3)
def matchings(pts):
    if not pts:
        return [[]]
    a = pts[0]; out = []
    for b in pts[1:]:
        rest = [p for p in pts if p not in (a, b)]
        out += [[(a, b)] + m for m in matchings(rest)]
    return out
counts = [len(matchings(list(range(2 * k)))) for k in range(1, 7)]
okc = counts == [prod(range(2 * k - 1, 0, -2)) for k in range(1, 7)] and \
      all(len(matchings(list(range(2 * k)))) == SetPartitions(2 * k, [2] * k).cardinality() for k in range(1, 5))
y = var("y"); oks = bool(integrate(y ** 6 * exp(-y ** 2 / 2), y, -oo, oo) / sqrt(2 * pi) == 15)
print("[PASS] the number of Wick pairings of 2n points is (2n-1)!! = 1, 3, 15, 105, 945, 10395 for n = 1..6, and <y^6> = 15 for a unit Gaussian:", okc and oks)
# 3. the thermal propagator by images and by Matsubara frequencies
w, b, t, kk = var("w b t kk")
assume(w > 0); assume(b > 0); assume(t > 0); assume(t < b)
images = (sum(exp(-w * (t + kk * b)), kk, 0, oo) + sum(exp(-w * (kk * b - t)), kk, 1, oo)) / (2 * w)
closed = cosh(w * (b / 2 - t)) / (2 * w * sinh(w * b / 2))
print("[PASS] sum over images sum_k e^{-omega|tau + k beta|}/(2 omega) = cosh(omega(beta/2 - tau))/(2 omega sinh(omega beta/2)) for 0 < tau < beta:",
      bool((images.exponentialize() - closed.exponentialize()).simplify_full() == 0))
nn = var("nn")
mats = sum(1 / ((2 * pi * nn / b) ** 2 + w ** 2), nn, -oo, oo) / b      # Sage returns digamma functions
nB = 1 / (exp(b * w) - 1)
okm = all(abs(mats.subs(w=wv, b=bv).n(200) - closed.subs(t=0, w=wv, b=bv).n(200)) < 1e-45 for wv, bv in [(1, 1), (1, 3), (QQ(1) / 2, 2), (3, QQ(1) / 4)]) \
      and bool(((nB + QQ(1) / 2) / w - closed.subs(t=0)).exponentialize().simplify_full() == 0)
print("[PASS] Matsubara sum (1/beta) sum_n 1/(omega_n^2 + omega^2) with omega_n = 2 pi n/beta equals coth(beta omega/2)/(2 omega) (to 45 digits at four points), which is exactly (n_B + 1/2)/omega:", okm)
print("[PASS] G_beta solves -G'' + omega^2 G = 0 on (0, beta), is periodic, and tends to e^{-omega tau}/(2 omega) as beta -> oo:",
      bool((-diff(closed, t, 2) + w ** 2 * closed).simplify_full() == 0) and bool(closed.subs(t=0) == closed.subs(t=b)) and
      bool(limit(closed.exponentialize(), b=oo) == exp(-w * t) / (2 * w)))
print("wrote", os.path.relpath(OUT))
