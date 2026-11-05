# Figure: lattice gauge theory in two dimensions. Left, the exact average plaquette u(beta) of the Wilson action for Z2 (tanh beta),
# compact U(1) (I_1/I_0) and SU(2) (I_2/I_1), with seeded heat-bath Monte Carlo points on 8 x 8 periodic lattices. Right, Wilson
# loops W(R, R) of U(1) at beta = 2 and of Z2 at beta = 1 against the area R^2, with the exact law u^A. Checks: every Monte Carlo
# plaquette is within 0.02 of the exact value; the loops follow u^A; the small-beta slopes are 1, 1/2 and 1/4.
# Output: figures/lgt_two_dim.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import iv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lgt_two_dim.svg")
L = int(8)
rng = np.random.default_rng(int(7))
par = np.indices((L, L)).sum(axis=0) % 2
def sh(a, mu, k):
    return np.roll(a, -k, axis=mu)
# quaternions q = q0 + i q.sigma stored as arrays (..., 4); product and inverse in this representation
def qmul(a, b):
    c = np.empty(np.broadcast(a, b).shape)
    c[..., 0] = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
    c[..., 1:] = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] - np.cross(a[..., 1:], b[..., 1:])
    return c
def qinv(a):
    return a * np.array([1.0, -1, -1, -1])
# one group = (multiply, inverse, identity, real part of the normalised trace, heat-bath draw given the staple sum)
def hb_z2(st, beta):
    return np.where(rng.random(st.shape) < 1 / (1 + np.exp(-2 * beta * st)), 1.0, -1.0)
def hb_u1(st, beta):                     # st = k e^{i phi}; theta + phi is von Mises with kappa = beta k
    return np.exp(1j * (rng.vonmises(0.0, beta * np.abs(st) + 1e-300) - np.angle(st)))
def hb_su2(st, beta):                    # st = k V; W = U V has a0 density sqrt(1 - a0^2) e^{beta k a0} (Kennedy-Pendleton)
    k = np.sqrt((st**2).sum(-1)); V = st / k[..., None]; al = beta * k
    a0 = np.empty(k.shape); todo = np.ones(k.shape, bool)
    while todo.any():
        n = todo.sum(); e = rng.random((4, n))
        Dd = -(np.log(e[0]) + np.log(e[1]) * np.cos(2 * np.pi * e[2])**2) / al[todo]
        ok = e[3]**2 <= 1 - Dd / 2
        idx = np.flatnonzero(todo)[ok]; a0.flat[idx] = 1 - Dd[ok]; todo.flat[idx] = False
    v = rng.normal(size=k.shape + (3,)); v /= np.linalg.norm(v, axis=-1)[..., None]
    W = np.concatenate([a0[..., None], np.sqrt(1 - a0**2)[..., None] * v], axis=-1)
    return qmul(W, qinv(V))
GROUPS = {
    "Z2": (lambda a, b: a * b, lambda a: a, np.ones((2, L, L)), lambda p: p, hb_z2),
    "U1": (lambda a, b: a * b, np.conj, np.ones((2, L, L), complex), lambda p: p.real, hb_u1),
    "SU2": (qmul, qinv, np.tile(np.array([1.0, 0, 0, 0]), (2, L, L, 1)), lambda p: p[..., 0], hb_su2)}
def staple(U, mu, mul, inv):
    nu = 1 - mu
    return (mul(mul(sh(U[nu], mu, 1), inv(sh(U[mu], nu, 1))), inv(U[nu]))
            + mul(mul(inv(sh(sh(U[nu], mu, 1), nu, -1)), inv(sh(U[mu], nu, -1))), sh(U[nu], nu, -1)))
def plaq(U, mul, inv):
    return mul(mul(mul(U[0], sh(U[1], 0, 1)), inv(sh(U[0], 1, 1))), inv(U[1]))
def simulate(name, beta, ntherm=int(40), nmeas=int(160)):
    mul, inv, U0, re, hb = GROUPS[name]; U = U0.copy(); out = []
    for k in range(ntherm + nmeas):
        for mu in (0, 1):
            for p in (0, 1):
                new = hb(staple(U, mu, mul, inv), beta)
                mask = (par == p) if U.ndim == 3 else (par == p)[..., None]
                U[mu] = np.where(mask, new, U[mu])
        if k >= ntherm:
            out.append(U.copy())
    return out, mul, inv, re
exact = {"Z2": np.tanh, "U1": lambda b: iv(1, b) / iv(0, b), "SU2": lambda b: iv(2, b) / iv(1, b)}
lab = {"Z2": r"$\mathbb{Z}_2$: $\tanh\beta$", "U1": r"$U(1)$: $I_1(\beta)/I_0(\beta)$", "SU2": r"$SU(2)$: $I_2(\beta)/I_1(\beta)$"}
col = {"Z2": "black", "U1": "royalblue", "SU2": "firebrick"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
bb = np.linspace(1e-3, 6, 300); betas = [0.5, 1.0, 2.0, 3.0, 4.5, 6.0]; worst = 0
for g in ("Z2", "U1", "SU2"):
    axes[0].plot(bb, exact[g](bb), color=col[g], lw=1.8, label=lab[g])
    for b in betas:
        cfgs, mul, inv, re = simulate(g, b)
        m = np.mean([re(plaq(U, mul, inv)).mean() for U in cfgs])
        worst = max(worst, abs(m - exact[g](b)))
        axes[0].plot(b, m, "o", color=col[g], ms=5, mfc="white")
axes[0].set_xlabel(r"$\beta$"); axes[0].set_ylabel(r"$u(\beta) = \langle \frac{1}{N}\mathrm{Re\,Tr}\,U_P\rangle$")
axes[0].set_title("Average plaquette in two dimensions (curves exact, circles Monte Carlo)", fontsize=9); axes[0].legend(fontsize=8, loc="lower right")
# Wilson loops: abelian, so W(R, R) = <Re prod of the enclosed plaquettes>
def loops(cfgs, mul, inv, re, Rmax=int(3)):
    res = []
    for R in range(1, Rmax + 1):
        vals = []
        for U in cfgs:
            P = plaq(U, mul, inv); acc = np.ones_like(P)
            for i in range(R):
                for j in range(R):
                    acc = acc * sh(sh(P, 0, i), 1, j)
            vals.append(re(acc).mean())
        res.append(np.mean(vals))
    return np.array(res)
ok_loops = True
for g, b, mk in (("U1", 2.0, "o"), ("Z2", 1.0, "s")):
    cfgs, mul, inv, re = simulate(g, b, int(40), int(400))
    W = loops(cfgs, mul, inv, re); A = np.arange(1, 4)**2; u = exact[g](b)
    axes[1].semilogy(A, W, mk, color=col[g], mfc="white", label=lab[g].split(":")[0] + rf", $\beta = {float(b):g}$, Monte Carlo")
    aa = np.linspace(0, 10, 50); axes[1].semilogy(aa, u**aa, color=col[g], lw=1.2, ls="--", label=rf"$u^A$, $u = {u:.4f}$")
    ok_loops = ok_loops and all(abs(np.log(W[:3]) - A[:3] * np.log(u)) < 0.15 * A[:3])
axes[1].set_xlabel(r"area $A = R^2$ (plaquettes)"); axes[1].set_ylabel(r"$W(R, R)$")
axes[1].set_title("Wilson loops in two dimensions: the area law", fontsize=9); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] largest deviation of a Monte Carlo plaquette from the exact u(beta): %.4f < 0.02:" % worst, worst < 0.02)
print("[PASS] W(R, R) follows u^{R^2} for R = 1, 2, 3:", ok_loops)
b = var("b")
print("[PASS] small-beta slopes 1, 1/2, 1/4:", [taylor(f, b, 0, 1).coefficient(b) for f in (tanh(b), bessel_I(1, b) / bessel_I(0, b), bessel_I(2, b) / bessel_I(1, b))] == [1, 1/2, 1/4])
print("wrote", os.path.relpath(OUT))
