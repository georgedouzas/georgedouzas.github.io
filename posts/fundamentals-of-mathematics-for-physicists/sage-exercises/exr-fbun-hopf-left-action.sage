# Exercise exr-fbun-hopf-left-action: the opposite phases (a e^{is}, b e^{-is}) move the image of the Hopf map by
# the rotation through 2s about the Z-axis, and the swap-type transformation (a, b) -> (cos(s) a - sin(s) b,
# sin(s) a + cos(s) b) rotates it through 2s about the Y-axis: unit quaternions acting on S^3 act on S^2 by rotations
# of twice the angle, the double cover SU(2) -> SO(3).
a1, a2, b1, b2, s = var("a1 a2 b1 b2 s", domain="real")
def hopf(a, b):
    ab = (a * b.conjugate()).expand()
    return vector([2 * ab.real(), 2 * ab.imag(), (a * a.conjugate() - b * b.conjugate()).expand().real()])
a, b = a1 + I * a2, b1 + I * b2
P = hopf(a, b)
Rz = matrix([[cos(2 * s), -sin(2 * s), 0], [sin(2 * s), cos(2 * s), 0], [0, 0, 1]])
Ry = matrix([[cos(2 * s), 0, sin(2 * s)], [0, 1, 0], [-sin(2 * s), 0, cos(2 * s)]])
P1 = hopf(a * exp(I * s), b * exp(-I * s))
print("pi(a e^{is}, b e^{-is}) = R_z(2s) pi(a, b):", bool(all(e.simplify_full().trig_reduce() == 0 for e in P1 - Rz * P)))
P2 = hopf(cos(s) * a - sin(s) * b, sin(s) * a + cos(s) * b)
diffs = [e.simplify_full().trig_reduce() for e in P2 - Ry * P]
print("pi(cos s a - sin s b, sin s a + cos s b) = R(2s) pi(a, b) about the Y-axis:", bool(all(e == 0 for e in diffs)))
print("s = pi gives (a, b) -> (-a, -b) on S^3 but the identity rotation:", Rz.subs(s=pi), sep="\n")
