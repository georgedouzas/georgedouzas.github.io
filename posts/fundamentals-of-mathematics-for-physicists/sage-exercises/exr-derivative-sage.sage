# Exercise exr-derivative-sage: derivatives by the rules and by Sage: the chain rule on sin(x^2), the product rule on
# x^2 e^x, the quotient rule on tan x, and the derivative of log x from the limit definition.
x, h = var("x h")
print("(sin x^2)' =", diff(sin(x ** 2), x), "; (x^2 e^x)' =", diff(x ** 2 * exp(x), x).factor(), "; (tan x)' =", diff(tan(x), x), "= 1/cos^2 x:", bool((diff(tan(x), x) - 1 / cos(x) ** 2).simplify_trig() == 0))
print("from the definition: lim (log(x + h) - log x)/h =", limit((log(x + h) - log(x)) / h, h=0))
print("second derivative of x^2 e^x:", diff(x ** 2 * exp(x), x, 2).factor())
print("the derivative of |x| at 0 does not exist: one-sided difference quotients", limit(abs(h) / h, h=0, dir="+"), "and", limit(abs(h) / h, h=0, dir="-"))
