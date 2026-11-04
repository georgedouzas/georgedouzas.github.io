# Exercise exr-kkr-modes-interval: modes on an interval instead of a circle. On 0 <= y <= pi R with Dirichlet
# conditions the eigenfunctions of -d_y^2 are sin(n y/R), n = 1, 2, ..., with eigenvalues (n/R)^2: masses n/R as on
# the circle but one state per level and no zero mode; with Neumann conditions cos(n y/R), n = 0, 1, ..., the zero
# mode returns.
y, R = var("y R"); assume(R > 0); n = var("n"); assume(n, "integer")
s = sin(n * y / R); c = cos(n * y / R)
print("[PASS] -d^2 sin(n y/R) = (n/R)^2 sin(n y/R) and sin vanishes at y = 0, pi R:",
      bool((-diff(s, y, 2) - (n / R)^2 * s).simplify_full() == 0) and bool(s.subs(y=0) == 0) and bool(s.subs(y=pi * R) == 0))
print("[PASS] -d^2 cos(n y/R) = (n/R)^2 cos(n y/R) and its derivative vanishes at y = 0, pi R:",
      bool((-diff(c, y, 2) - (n / R)^2 * c).simplify_full() == 0) and bool(diff(c, y).subs(y=0) == 0) and bool(diff(c, y).subs(y=pi * R) == 0))
print("[PASS] orthogonality int_0^{pi R} sin(2y/R) sin(3y/R) dy = 0 and int sin^2(2y/R) = pi R/2:",
      bool(integrate(sin(2 * y / R) * sin(3 * y / R), y, 0, pi * R) == 0) and bool(integrate(sin(2 * y / R)^2, y, 0, pi * R) == pi * R / 2))
