# Exercise exr-gth-orbit-sphere: in the SO(3) vector model V = (lam/4)(phi.phi - a^2)^2 the rotation
# R = R_z(alpha) R_y(beta) R_z(gamma) maps the vacuum (0, 0, a) to a (sin beta cos alpha, sin beta sin alpha, cos beta):
# the orbit is the whole sphere of radius a, and every point of it is again a vacuum.
al, be, ga, a, lam = var("al be ga a lam", domain="real")
Rz = lambda u: matrix(SR, [[cos(u), -sin(u), 0], [sin(u), cos(u), 0], [0, 0, 1]])
Ry = lambda u: matrix(SR, [[cos(u), 0, sin(u)], [0, 1, 0], [-sin(u), 0, cos(u)]])
R = Rz(al) * Ry(be) * Rz(ga)
print("[PASS] R is orthogonal with determinant 1:", (R.transpose() * R).simplify_full() == 1 and bool(R.det().simplify_full() == 1))
v0 = vector(SR, [0, 0, a])
w = (R * v0).simplify_full()
print("R (0, 0, a) =", w)
print("[PASS] the image is a (sin b cos a, sin b sin a, cos b), the sphere in spherical coordinates:", bool(w == a * vector([sin(be) * cos(al), sin(be) * sin(al), cos(be)])))
p = vector(SR, var("p1 p2 p3", domain="real"))
V = lambda f: lam / 4 * (f * f - a^2)^2
print("[PASS] V(R phi) = V(phi) for every phi (V is SO(3)-invariant):", bool((V(R * p) - V(p)).simplify_full() == 0))
print("[PASS] V vanishes on the whole orbit:", bool(V(w).simplify_full() == 0))
print("[PASS] gamma drops out: rotations about the vacuum axis fix it (the stabiliser SO(2)):", bool(diff(w, ga) == 0))
