# Exercise exr-mean-value-sage: the mean value theorem for x^3 on [0, 2], sin x on [0, pi] and e^x on [0, 1]: the
# points c where f'(c) equals the average slope (f(b) - f(a))/(b - a), found by solving the equation.
x = var("x")
cases = [(x ** 3, 0, 2, 2 / sqrt(3)), (sin(x), 0, pi, pi / 2), (exp(x), 0, 1, log(e - 1))]
for f, a, b, c in cases:
    slope = ((f.subs(x=b) - f.subs(x=a)) / (b - a)).simplify_full()
    print("f =", f, "on [", a, ",", b, "]: average slope", slope, "; c =", c, "=", c.n(digits=6), "lies in the interval:", bool(a < c < b), "; f'(c) = slope:", bool((diff(f, x).subs(x=c) - slope).simplify_full() == 0))
print("solve f'(x) = 4 for f = x^3:", solve(3 * x ** 2 == 4, x))
