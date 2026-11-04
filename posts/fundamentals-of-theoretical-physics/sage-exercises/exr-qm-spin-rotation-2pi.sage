# Exercise exr-qm-spin-rotation-2pi: rotations exp(-i theta S_2/hbar) for spin 1/2 and spin 1 at 2 pi and 4 pi.
th = var("th", domain="real")
s2 = matrix(SR, [[0, -I], [I, 0]]) / 2                         # S_2/hbar for spin 1/2
J2 = matrix(SR, [[0, -I, 0], [I, 0, -I], [0, I, 0]]) / sqrt(2)   # S_2/hbar for spin 1
for name, M in [("spin 1/2", s2), ("spin 1", J2)]:
    R = (-I * th * M).exp().apply_map(lambda e: e.exponentialize().simplify_full())
    print(name, ": R(2 pi) =", R.subs(th=2 * pi).simplify_full().list(), " R(4 pi) =", R.subs(th=4 * pi).simplify_full().list())
R = (-I * th * s2).exp().apply_map(lambda e: e.rectform().trig_reduce().simplify_full())
print("spin 1/2: R(theta) ="); print(R)
print("R(pi/2) |up> =", (R.subs(th=pi / 2) * vector(SR, [1, 0])).simplify_full(), "= |+x> = (1, 1)/sqrt 2")
