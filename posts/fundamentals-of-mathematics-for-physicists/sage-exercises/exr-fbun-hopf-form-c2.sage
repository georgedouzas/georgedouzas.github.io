# Exercise exr-fbun-hopf-form-c2: the one-form alpha = Im(conj(a) da + conj(b) db) = x1 dx2 - x2 dx1 + x3 dx4 - x4 dx3
# on C^2 = R^4, pulled back to SU(2) by the Euler-angle parametrisation of the Hopf map's total space, is
# -(1/2) sigma^3 = -(1/2)(d psi + cos(theta) d phi): the Hopf connection is (up to the factor) the restriction of a
# form defined on all of C^2.
R4 = Manifold(4, "R4"); C4 = R4.chart("x1 x2 x3 x4"); x1, x2, x3, x4 = C4[:]
alpha = R4.one_form(-x2, x1, -x4, x3, name="alpha")
G = Manifold(3, "SU2"); U = G.open_subset("U")
E = U.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi"); th, ph, ps = E[:]
a = cos(th / 2) * exp(-I * (ph + ps) / 2); b = -I * sin(th / 2) * exp(I * (ph - ps) / 2)
emb = U.diff_map(R4, {(E, C4): [a.real().simplify_full(), a.imag().simplify_full(), b.real().simplify_full(), b.imag().simplify_full()]}, name="emb")
pa = emb.pullback(alpha)
pa.apply_map(lambda e: e.simplify_full().trig_reduce())
print("emb^* alpha =", pa.display())
s3 = U.one_form(0, cos(th), 1, name="sigma3")
print("emb^* alpha = -(1/2) sigma3:", pa == -s3 / 2)
print("d(emb^* alpha) =", pa.exterior_derivative().display())
print("on a fibre (theta, phi fixed) the form is -(1/2) d psi; psi runs over [0, 4 pi), so it integrates to",
      integrate(-1 / 2, ps, 0, 4 * pi), "= -2 pi, the length of the great circle up to sign")
