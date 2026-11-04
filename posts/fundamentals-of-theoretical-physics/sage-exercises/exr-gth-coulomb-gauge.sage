# Exercise exr-gth-coulomb-gauge: bring A = (t x^2, x y, 0), Phi = 0 to the Coulomb gauge div A = 0 by the gauge
# transformation A -> A + grad lambda, Phi -> Phi - d_t lambda, and check that E = -grad Phi - d_t A and B = curl A
# are unchanged.
t, x, y, z = var("t x y z", domain="real")
Av = vector(SR, [t * x^2, x * y, 0]); Phi = SR(0)
div = lambda W: sum(diff(W[i], [x, y, z][i]) for i in range(3))
grad = lambda f: vector(SR, [diff(f, w) for w in (x, y, z)])
curl = lambda W: vector(SR, [diff(W[2], y) - diff(W[1], z), diff(W[0], z) - diff(W[2], x), diff(W[1], x) - diff(W[0], y)])
E = lambda P, W: -grad(P) - diff(W, t)
print("div A =", div(Av))
lam = -t * x^3 / 3 - x^3 / 6
print("lambda =", lam, "; its Laplacian =", div(grad(lam)), "= -div A")
A2 = Av + grad(lam); Phi2 = Phi - diff(lam, t)
print("new potentials: A' =", A2, "; Phi' =", Phi2)
print("[PASS] the new vector potential is divergence-free:", bool(div(A2) == 0))
print("[PASS] E and B are unchanged:", bool(E(Phi2, A2) == E(Phi, Av)) and bool(curl(A2) == curl(Av)))
print("E =", E(Phi, Av), "; B =", curl(Av))
