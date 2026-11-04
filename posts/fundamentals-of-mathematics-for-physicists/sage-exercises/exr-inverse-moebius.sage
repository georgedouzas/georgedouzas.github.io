# Exercise exr-inverse-moebius: the inverse of (x - 1)/(x + 1).
x, y = var("x y")
f = (x - 1) / (x + 1)
finv = solve(f == y, x)[0].rhs()
print("f^{-1}(y) =", finv)
print("[PASS] f(f^{-1}(y)) = y and f^{-1}(f(x)) = x:", bool(f.subs(x=finv).simplify_full() == y) and bool(finv.subs(y=f).simplify_full() == x))
