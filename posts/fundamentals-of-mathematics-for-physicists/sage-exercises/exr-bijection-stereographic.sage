# Exercise exr-bijection-stereographic: stereographic projection (X, Y, Z) -> (X, Y)/(1 - Z) from the sphere minus
# the north pole onto the plane, and its inverse (u, v) -> (2u, 2v, u^2 + v^2 - 1)/(u^2 + v^2 + 1).
X, Y, Z, u, v = var("X Y Z u v")
proj = lambda P: vector([P[0] / (1 - P[2]), P[1] / (1 - P[2])])
inv = lambda q: vector([2 * q[0], 2 * q[1], q[0]^2 + q[1]^2 - 1]) / (q[0]^2 + q[1]^2 + 1)
back = proj(inv(vector([u, v]))).apply_map(lambda e: e.simplify_full())
print("proj(inv(u, v)) =", back)
print("[PASS] inv(u, v) lies on the sphere:", bool((inv(vector([u, v])).norm()^2 - 1).simplify_full() == 0))
# on the sphere, parametrise by (theta, phi) to check inv(proj(P)) = P
th, ph = var("th ph")
P = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
forth = inv(proj(P)).apply_map(lambda e: e.simplify_full())
print("inv(proj(P)) - P =", (forth - P).apply_map(lambda e: e.simplify_full()))
print("[PASS] both compositions are the identity:", back == vector([u, v]) and all(bool(e == 0) for e in (forth - P).apply_map(lambda e: e.simplify_full())))
