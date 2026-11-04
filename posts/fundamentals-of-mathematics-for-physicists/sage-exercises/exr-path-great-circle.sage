# Exercise exr-path-great-circle: a great-circle path from the north pole to (1, 0, 0).
t = var("t")
g = vector([sin(pi * t / 2), 0, cos(pi * t / 2)])
print("gamma(0) =", g.subs(t=0), ", gamma(1) =", g.subs(t=1), ", |gamma|^2 =", (g * g).simplify_full())
print("[PASS] the path starts at the pole, ends at (1,0,0) and stays on the sphere:", g.subs(t=0) == vector([0, 0, 1]) and g.subs(t=1) == vector([1, 0, 0]) and bool((g * g).simplify_full() == 1))
