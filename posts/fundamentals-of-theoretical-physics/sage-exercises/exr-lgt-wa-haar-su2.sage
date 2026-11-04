# Exercise exr-lgt-wa-haar-su2: moments of a0 = (1/2) Tr U under the Haar measure of SU(2), exactly and by sampling S^3.
import numpy as np
psi = var('psi')
w = 2/pi * sin(psi)^2
print('normalisation:', integrate(w, psi, 0, pi))
print('<a0^2> =', integrate(w * cos(psi)^2, psi, 0, pi), ' <a0^4> =', integrate(w * cos(psi)^4, psi, 0, pi))
rng = np.random.default_rng(int(11))
v = rng.normal(size=(int(400000), int(4))); v /= np.linalg.norm(v, axis=1)[:, None]   # uniform on S^3
print('sampled <a0> = %.4f, <a0^2> = %.4f, <a0^4> = %.4f' % (v[:, 0].mean(), (v[:, 0]**2).mean(), (v[:, 0]**4).mean()))
print('sampled <(1/2 Tr U)^2> = <a0^2> = 1/4 means <|Tr U|^2> = 1 (Schur orthogonality)')
