# Exercise exr-lgt-st-code: the change of the Wilson action under a single-link update, from the staple and from the full action.
import numpy as np
rng = np.random.default_rng(int(71)); L, D, beta = int(3), int(3), 2.0
def rsu2(*sh):
    v = rng.normal(size=sh + (4,)); return v / np.linalg.norm(v, axis=-1)[..., None]
def qmul(a, b):
    c = np.empty(np.broadcast(a, b).shape)
    c[..., 0] = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
    c[..., 1:] = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] - np.cross(a[..., 1:], b[..., 1:])
    return c
qinv = lambda a: a * np.array([1.0, -1, -1, -1])
sh = lambda a, mu, k=1: np.roll(a, -k, axis=mu)
def action(U):
    S = 0
    for m in range(D):
        for n in range(m + 1, D):
            P = qmul(qmul(qmul(U[m], sh(U[n], m)), qinv(sh(U[m], n))), qinv(U[n]))
            S += beta * (1 - P[..., 0]).sum()
    return S
def staple(U, mu):
    return sum(qmul(qmul(sh(U[nu], mu), qinv(sh(U[mu], nu))), qinv(U[nu]))
               + qmul(qmul(qinv(sh(sh(U[nu], mu), nu, -1)), qinv(sh(U[mu], nu, -1))), sh(U[nu], nu, -1)) for nu in range(D) if nu != mu)
U = rsu2(D, L, L, L); mu, x = 1, (0, 2, 1)
new = rsu2(); V = U.copy(); V[(mu,) + x] = new
Sig = staple(U, mu)[x]
dS_staple = -beta * (qmul(new, Sig)[0] - qmul(U[(mu,) + x], Sig)[0])
print('Delta S from the full action: %.12f' % (action(V) - action(U)))
print('Delta S from the staple:      %.12f' % dS_staple)
print('number of staples per link: 2(d - 1) =', 2 * (D - 1))
