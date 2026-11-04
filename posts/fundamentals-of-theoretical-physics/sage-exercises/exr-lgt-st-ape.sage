# Exercise exr-lgt-st-ape: APE smearing U -> P_SU(2)(U + zeta Sigma^dagger) of a random 2D SU(2) configuration, eight steps of zeta = 0.3.
import numpy as np
rng = np.random.default_rng(int(81)); L = int(8)
def qmul(a, b):
    c = np.empty(np.broadcast(a, b).shape)
    c[..., 0] = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
    c[..., 1:] = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] - np.cross(a[..., 1:], b[..., 1:])
    return c
qinv = lambda a: a * np.array([1.0, -1, -1, -1])
sh = lambda a, mu, k=1: np.roll(a, -k, axis=mu)
def staple(U, mu):
    nu = 1 - mu
    return qmul(qmul(sh(U[nu], mu), qinv(sh(U[mu], nu))), qinv(U[nu])) + qmul(qmul(qinv(sh(sh(U[nu], mu), nu, -1)), qinv(sh(U[mu], nu, -1))), sh(U[nu], nu, -1))
plaq = lambda U: qmul(qmul(qmul(U[0], sh(U[1], 0)), qinv(sh(U[0], 1))), qinv(U[1]))[..., 0].mean()
U = rng.normal(size=(2, L, L, 4)); U /= np.linalg.norm(U, axis=-1)[..., None]
print('step 0: average (1/2) Tr U_P = %+.4f' % plaq(U))
for k in range(1, 9):
    V = np.stack([U[mu] + 0.3 * qinv(staple(U, mu)) for mu in (0, 1)])      # Sigma^dagger is the quaternion conjugate
    U = V / np.linalg.norm(V, axis=-1)[..., None]                             # projection back to SU(2)
    print('step %d: average (1/2) Tr U_P = %+.4f' % (k, plaq(U)))
