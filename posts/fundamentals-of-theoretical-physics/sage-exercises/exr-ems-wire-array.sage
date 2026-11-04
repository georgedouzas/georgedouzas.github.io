# Exercise exr-ems-wire-array (Macchi et al., Problem 1.12): phi = -C log(2 (cosh(k y) - cos(k x))) for wires at (n a, 0).
x, y, k, C, a, lam, s = var('x y k C a lam s'); assume(k > 0, a > 0, s > 0, lam > 0, y > 0)
phi = -C * log(2 * (cosh(k * y) - cos(k * x)))
print("two-dimensional Laplacian:", (diff(phi, x, 2) + diff(phi, y, 2)).simplify_full())
# near the wire at the origin: 2 (cosh ky - cos kx) ~ k^2 (x^2 + y^2)
print("expansion of 2 (cosh ky - cos kx) at small x, y:", (2 * (cosh(k * y) - cos(k * x))).taylor((x, 0), (y, 0), 2))
# matching -2 C log s = -(lam/2 pi) log s gives C = lam/(4 pi); far field: |E_y| = C k
Cval = lam / (4 * pi); kval = 2 * pi / a
print("C k with C = lam/(4 pi), k = 2 pi/a:", (Cval * kval).simplify_full(), "= sigma/2 with sigma = lam/a")
print("large-y slope of phi:", limit(diff(phi, y).subs(C=Cval, k=kval), y=oo))
