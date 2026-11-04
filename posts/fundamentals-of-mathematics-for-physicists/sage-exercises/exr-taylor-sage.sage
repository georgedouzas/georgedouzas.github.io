# Exercise exr-taylor-sage: Taylor polynomials of exp, log(1 + x), (1 + x)^(1/2) and 1/(1 - x) at 0, and the
# remainder of the degree-4 polynomial of e^x on [0, 1] compared with the Lagrange bound e/5!.
x = var("x")
for f in [exp(x), log(1 + x), sqrt(1 + x), 1 / (1 - x)]:
    print("T_4 of", f, ":", taylor(f, x, 0, 4))
T4 = taylor(exp(x), x, 0, 4)
err = max(abs(exp(t) - T4.subs(x=t)).n() for t in [k / 20 for k in range(21)])
print("max |e^x - T_4(x)| on [0, 1] (sampled):", err, "<= e/5! =", (e / factorial(5)).n(), ":", bool(err <= (e / factorial(5)).n()))
print("Taylor polynomial of sin x at pi/2 to degree 4:", taylor(sin(x), x, pi / 2, 4))
