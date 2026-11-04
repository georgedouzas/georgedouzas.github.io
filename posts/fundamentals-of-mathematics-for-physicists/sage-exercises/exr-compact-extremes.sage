# Exercise exr-compact-extremes: extremes of x e^{-x} on [0, 2] and on (0, 2).
x = var("x"); f = x * exp(-x)
crit = solve(diff(f, x) == 0, x)
print("critical points:", crit, "; f(1) =", f.subs(x=1), "; f(0) =", f.subs(x=0), "; f(2) =", f.subs(x=2))
print("[PASS] maximum e^{-1} at x = 1, minimum 0 at the endpoint 0 (not attained on the open interval, where f > 0):",
      crit == [x == 1] and bool(f.subs(x=1) == exp(-1)) and bool(f.subs(x=0) == 0) and bool(f.subs(x=0.5) > 0))
