# Exercise exr-function-limits-sage: limits of functions with Sage's limit, including one-sided limits and limits
# that do not exist: sin(x)/x -> 1, (1 - cos x)/x^2 -> 1/2, |x|/x has different one-sided limits at 0, sin(1/x)
# has no limit at 0.
x = var("x")
print("lim_{x->0} sin(x)/x =", limit(sin(x) / x, x=0), "; lim (1 - cos x)/x^2 =", limit((1 - cos(x)) / x ** 2, x=0), "; lim (e^x - 1)/x =", limit((exp(x) - 1) / x, x=0))
print("one-sided limits of |x|/x at 0: from the right", limit(abs(x) / x, x=0, dir="+"), ", from the left", limit(abs(x) / x, x=0, dir="-"))
print("sin(1/x) at 0:", limit(sin(1 / x), x=0), "(Sage reports an indeterminate value: the function oscillates between -1 and 1)")
print("limits at infinity: lim x e^{-x} =", limit(x * exp(-x), x=oo), "; lim (log x)/x =", limit(log(x) / x, x=oo), "; lim x^2/e^x =", limit(x ** 2 / exp(x), x=oo))
