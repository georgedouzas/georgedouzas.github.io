# Exercise exr-continuity-check: continuity of piecewise functions at the junction. f(x) = x^2 for x < 1 and 2x - 1
# for x >= 1 is continuous at 1 (the one-sided limits agree with f(1)); g(x) = x^2 for x < 1 and 2x for x >= 1 is not.
x = var("x")
for name, left_piece, right_piece in [("f", x ** 2, 2 * x - 1), ("g", x ** 2, 2 * x)]:
    left = limit(left_piece, x=1, dir="-"); right = limit(right_piece, x=1, dir="+"); value = right_piece.subs(x=1)
    print(name, ": left limit", left, ", right limit", right, ", value at 1:", value, "-> continuous at 1:", bool(left == right == value))
print("f is also differentiable at 1: left derivative", diff(x ** 2, x).subs(x=1), ", right derivative", diff(2 * x - 1, x), "(equal, so the graph has no corner)")
print("|x| is continuous but not differentiable at 0: the difference quotients |h|/h have one-sided limits", limit(abs(x) / x, x=0, dir="+"), "and", limit(abs(x) / x, x=0, dir="-"))
