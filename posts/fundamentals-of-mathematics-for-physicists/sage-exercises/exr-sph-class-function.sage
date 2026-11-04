# Exercise exr-sph-class-function: expansion of class functions on SU(2) in characters, with coefficients
# (1/pi) int f chi_j sin^2(theta/2) dtheta: cos theta, cos^2(theta/2) (finite) and theta(2 pi - theta) (infinite).
th = var("theta")
chi = lambda j: sum(cos(m * th) for m in [j - i for i in range(int(2 * j) + 1)])
w = sin(th / 2) ** 2 / pi
c = lambda f, j: integrate(f * chi(j) * w, th, 0, 2 * pi)
js = [0, 1/2, 1, 3/2, 2]
print("cos theta:        ", [c(cos(th), j) for j in js])
print("cos^2(theta/2):   ", [c(cos(th / 2) ** 2, j) for j in js])
q = th * (2 * pi - th)
cq = [c(q, j) for j in js]
print("theta(2pi - theta):", cq, "; numerically", [v.n(digits=5) for v in cq])
print("the coefficients for integer j are -2(2j + 1)/(j^2 (j + 1)^2):", [bool(c(q, j) == -2 * (2 * j + 1) / (j ** 2 * (j + 1) ** 2)) for j in [1, 2, 3, 4]], "; j^3 times the coefficient for j = 2, 4, 8:", [(j ** 3 * c(q, j)).n(digits=4) for j in [2, 4, 8]])
