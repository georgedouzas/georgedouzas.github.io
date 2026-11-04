# Exercise exr-product-torus-point: a point of the torus from two angles, in R^4 and as a surface of revolution.
u, v = pi / 3, pi / 2
P4 = vector([cos(u), sin(u), cos(v), sin(v)])
P3 = vector([(2 + cos(v)) * cos(u), (2 + cos(v)) * sin(u), sin(v)])
print("in R^4:", P4, "; in R^3:", P3)
print("[PASS] (1/2, sqrt3/2, 0, 1) and (1, sqrt3, 1):", P4 == vector([1/2, sqrt(3)/2, 0, 1]) and P3 == vector([1, sqrt(3), 1]))
