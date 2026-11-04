# Exercise exr-closed-forms-logistic-family: y = 1/(1 + C e^(-x)) solves y' = y(1 - y) for every C; C = 1 gives
# y(0) = 1/2, C = 0 the constant solution 1, and C -> oo the constant solution 0.
x, C = var("x C")
y = 1 / (1 + C * exp(-x))
res = (diff(y, x) - y * (1 - y)).simplify_full()
print("residual for general C:", res, "-> zero:", bool(res == 0))
print("C = 1: y(0) =", y.subs(C=1, x=0))
print("C = 0: y =", y.subs(C=0))
print("C -> infinity: y ->", limit(y, C=oo))
