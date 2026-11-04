# Exercise exr-radius-ratio: radii of convergence from the ratio test, R = lim |a_n/a_{n+1}|, for sum n! x^n (R = 0),
# sum x^n/n^2 (R = 1), sum 2^n x^n (R = 1/2) and sum x^n/n! (R = oo).
n = var("n")
for a, name in [(factorial(n), "n! x^n"), (1 / n ** 2, "x^n/n^2"), (2 ** n, "2^n x^n"), (1 / factorial(n), "x^n/n!")]:
    R = limit(abs(a / a.subs(n=n + 1)), n=oo)
    print("series", name, ": R = lim |a_n/a_{n+1}| =", R)
x = var("x")
print("check: sum x^n/n^2 at x = 1 is pi^2/6 (converges on the boundary):", sum(1 / n ** 2, n, 1, oo), "; sum 2^n x^n at x = 1/2 is the divergent sum of 1s")
