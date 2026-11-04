# Exercise exr-ls-radius-sage: the Laplace-Beltrami operator of SageManifolds on the sphere of radius R and on the torus with
# radii R1, R2, and the eigenvalue equations with symbolic radii: l(l + 1)/R^2 and m^2/R1^2 + n^2/R2^2.
R, R1, R2 = var("R R1 R2", domain="positive")
S2 = Manifold(2, "S2", structure="Riemannian")
X.<th, ph> = S2.chart(r"th:(0,pi):\theta ph:(0,2*pi):\phi")
g = S2.metric(); g[0, 0] = R ** 2; g[1, 1] = R ** 2 * sin(th) ** 2
print("sphere metric:", g.display())
is_zero = lambda e: bool(e.simplify_full().canonicalize_radical().simplify_full() == 0)
for l in range(3):
    Y = spherical_harmonic(l, 0, th, ph)
    print("l = %d: Laplacian of Y_l0 =" % l, S2.scalar_field(Y).laplacian().expr().simplify_full().canonicalize_radical(), "; = -l(l+1)/R^2 Y_lm for all m:",
          all(is_zero(S2.scalar_field(spherical_harmonic(l, m, th, ph)).laplacian().expr() + l * (l + 1) / R ** 2 * spherical_harmonic(l, m, th, ph)) for m in range(-l, l + 1)))
T2 = Manifold(2, "T2", structure="Riemannian")
Z.<a, b> = T2.chart(r"a:(0,2*pi):\theta b:(0,2*pi):\phi")
gT = T2.metric(); gT[0, 0] = R1 ** 2; gT[1, 1] = R2 ** 2
print("torus metric:", gT.display())
m, n = 2, 3
print("Laplacian of e^{i(2 theta + 3 phi)} =", T2.scalar_field(exp(I * (m * a + n * b))).laplacian().expr().simplify_full())
print("= -(m^2/R1^2 + n^2/R2^2) e^{i(m theta + n phi)} for |m|, |n| <= 3:", all(is_zero(T2.scalar_field(exp(I * (m * a + n * b))).laplacian().expr() + (m ** 2 / R1 ** 2 + n ** 2 / R2 ** 2) * exp(I * (m * a + n * b))) for m in range(-3, 4) for n in range(-3, 4)))
