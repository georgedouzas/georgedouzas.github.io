# Exercise exr-clm-phase-vector-field: Hamiltonian vector fields of the pendulum and of the free particle on the sphere.
q, p, th, ph, pth, pph = var("q p theta phi p_theta p_phi", domain="real")
def XH(H, qs, ps): return [diff(H, a) for a in ps] + [-diff(H, a) for a in qs]   # components along (q..., p...)
def apply(X, f, qs, ps): return sum(c * diff(f, v) for c, v in zip(X, list(qs) + list(ps)))
H1 = p**2 / 2 - cos(q)
X1 = XH(H1, [q], [p]); print("pendulum X_H components (d_q, d_p):", X1, "; X_H(H) =", apply(X1, H1, [q], [p]))
H2 = (pth**2 + pph**2 / sin(th)**2) / 2
X2 = XH(H2, [th, ph], [pth, pph]); print("sphere X_H components (d_theta, d_phi, d_p_theta, d_p_phi):", [c.simplify_full() for c in X2])
print("X_H(H) =", apply(X2, H2, [th, ph], [pth, pph]).simplify_full(), "; X_H(p_phi) =", apply(X2, pph, [th, ph], [pth, pph]))
