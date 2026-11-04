# Exercise exr-limit-squeeze: the squeeze theorem on n sin(1/n) and on (sin n)/n, with the bounding sequences, and
# Sage's limit for comparison.
n = var("n")
print("cos(1/n) <= n sin(1/n) <= 1 for n >= 1 (since x cos x <= sin x <= x for 0 <= x <= 1); both bounds -> 1:", limit(cos(1 / n), n=oo), limit(n * sin(1 / n), n=oo))
print("-1/n <= sin(n)/n <= 1/n, so sin(n)/n -> 0:", limit(sin(n) / n, n=oo))
print("numerical check for n = 10^6: n sin(1/n) =", numerical_approx(10 ** 6 * sin(1 / 10 ** 6), digits=15), "; sin(n)/n =", numerical_approx(sin(10 ** 6) / 10 ** 6, digits=6))
