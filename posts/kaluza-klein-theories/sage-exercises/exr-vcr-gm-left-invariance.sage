# Exercise exr-vcr-gm-left-invariance: the left-invariant forms of the Heisenberg group, sigma^y = dy,
# sigma^1 = dz1 + m y dz2, sigma^2 = dz2, are invariant under the identifications
# (y, z1, z2) -> (y + alpha, z1 - m alpha z2 + beta1, z2 + beta2) of Hull-Reid-Edwards, so they descend to the nilmanifold.
N = Manifold(3, "Nil"); X = N.chart("y z1 z2"); y, z1, z2 = X[:]
m, al, b1, b2 = var("m alpha beta1 beta2")
F = N.diff_map(N, [y + al, z1 - m * al * z2 + b1, z2 + b2], name="h")
sy = N.one_form(1, 0, 0, name="sigma_y"); s1 = N.one_form(0, 1, m * y, name="sigma_1"); s2 = N.one_form(0, 0, 1, name="sigma_2")
for s in (sy, s1, s2):
    print(s.display(), "  pulled back:", F.pullback(s).display(), "  invariant:", F.pullback(s) == s)
print("the naive form dz1 is not invariant:", F.pullback(N.one_form(0, 1, 0)).display())
