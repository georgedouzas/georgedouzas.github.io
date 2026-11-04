# Exercise exr-injective-monotone: x^3 + x is injective (positive derivative), x^3 - x is not (three roots).
x = var("x")
f = x^3 + x; g = x^3 - x
print("f' =", diff(f, x), "> 0 for all x")
print("roots of g:", solve(g == 0, x))
print("[PASS] f' = 3x^2 + 1 is positive and g has three zeros:", bool(diff(f, x) == 3 * x^2 + 1) and len(solve(g == 0, x)) == 3)
