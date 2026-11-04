# Exercise exr-ren-sphere-heat: the heat trace of the Laplacian on the unit sphere, sum (2l + 1) e^(-t l(l + 1)), fitted
# to c0 + c1 t + c2 t^2 at three small t.
R = RealField(100)
def K(t):
    t = R(t); return sum((2 * l + 1) * exp(-t * l * (l + 1)) for l in range(0, 3000))
ts = [R(0.01), R(0.02), R(0.05)]
A = matrix(R, [[1, t, t^2] for t in ts]); b = vector(R, [t * K(t) for t in ts])
c = A.solve_right(b)
print("fit: c0 = %.6f, c1 = %.6f, c2 = %.5f   (predicted 1, 1/3 = 0.333333, 1/15 = 0.06667)" % tuple(c))
print("zeta(0) = a_2 - (number of zero modes) =", QQ(1/3) - 1)
