# Exercise exr-clm-lrl-orbit: A.x = L^2 - m k r for the Laplace-Runge-Lenz vector A = p x L - m k x/r, which gives
# the conic r = (L^2/mk)/(1 + e cos phi) with e = |A|/(mk).
x, y, z, px, py, pz = var("x y z px py pz", domain="real")
m, k = var("m k", domain="positive")
X = vector([x, y, z]); P = vector([px, py, pz]); r = sqrt(X * X)
Lv = X.cross_product(P); A = P.cross_product(Lv) - m * k * X / r
print("A.x - (L^2 - m k r) =", (A * X - (Lv * Lv - m * k * r)).simplify_full())
# an example: pericentre at (1/(1+e), 0, 0) with p = 1 (m = k = L = 1); A points along +x with length e
e = var("e", domain="positive")
st = {x: 1 / (1 + e), y: 0, z: 0, px: 0, py: 1 + e, pz: 0, m: 1, k: 1}
print("A at the pericentre of the orbit with p = 1:", A.subs(st).simplify_full())
