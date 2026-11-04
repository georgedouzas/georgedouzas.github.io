# Exercise exr-csd-ppot-hessian: the Hessian of V = r^2/2 - r + 2/3, r = x_1^2 + x_2^2 + x_3^2 + x_4^2
# (Phi = (x_1 + i x_2, x_3 + i x_4)), at the vacuum (0, 0, 1, 0): one massive (radial) and three flat directions.
x = vector(SR, var("x1 x2 x3 x4", domain="real"))
r = x * x; V = r^2 / 2 - r + 2/3
H = matrix(SR, 4, 4, lambda i, j: diff(V, x[i], x[j]))
vac = {x[0]: 0, x[1]: 0, x[2]: 1, x[3]: 0}
Hv = H.subs(vac)
print("Hessian at the vacuum:"); print(Hv)
print("[PASS] gradient zero at the vacuum, Hessian eigenvalues 0, 0, 0, 4 with the non-zero one along the radial direction x_3:",
      all(bool(diff(V, v).subs(vac) == 0) for v in x) and sorted(Hv.eigenvalues()) == [0, 0, 0, 4] and bool(Hv[2, 2] == 4))
print("[PASS] the flat directions are tangent to the vacuum sphere r = 1: the Hessian vanishes on vectors orthogonal to the vacuum:",
      all((Hv * v).is_zero() for v in (vector([1, 0, 0, 0]), vector([0, 1, 0, 0]), vector([0, 0, 0, 1]))))
