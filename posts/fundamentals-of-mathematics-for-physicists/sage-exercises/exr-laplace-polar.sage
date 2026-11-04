# Exercise exr-laplace-polar: r^n cos(n theta), r^(-n) cos(n theta) and log r are harmonic for the polar Laplacian;
# r^n cos(m theta) is not for m != n.
r, th, n, m = var("r theta n m")
assume(n > 0)
lap = lambda u: (diff(u, r, 2) + diff(u, r) / r + diff(u, th, 2) / r ** 2).simplify_full()
for u in [r ** n * cos(n * th), r ** (-n) * cos(n * th), log(r)]:
    print("Laplacian of", u, "=", lap(u))
res = lap(r ** n * cos(m * th)).factor()
print("Laplacian of r^n cos(m theta) =", res, "-> vanishes only for m^2 = n^2")
