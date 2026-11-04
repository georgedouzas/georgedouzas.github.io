# Exercise exr-ems-A-to-coulomb: a gauge transformation bringing A = (x y, 0, 0) to the Coulomb gauge.
x, y, z = var('x y z')
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)])
def div(F): return sum(diff(F[i], v) for i, v in enumerate([x, y, z]))
A = vector([x * y, 0, 0]); lam = -y^3 / 6
Ap = A + vector([diff(lam, v) for v in [x, y, z]])
print("div A =", div(A), "; Laplacian of lambda =", sum(diff(lam, v, 2) for v in [x, y, z]))
print("A' =", Ap, "; div A' =", div(Ap), "; curl A =", curl(A), "; curl A' =", curl(Ap))
