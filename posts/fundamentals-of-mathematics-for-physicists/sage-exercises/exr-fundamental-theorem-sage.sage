# Exercise exr-fundamental-theorem-sage: the fundamental theorem of calculus in both directions: the derivative of
# F(x) = int_0^x e^{-t^2} dt is e^{-x^2}, and definite integrals are evaluated from antiderivatives F(b) - F(a).
x, t = var("x t")
assume(x > 0)
F = integrate(exp(-t ** 2), t, 0, x)
print("F(x) = int_0^x e^{-t^2} dt =", F, "; F'(x) =", diff(F, x).simplify_full())
for f, a, b in [(sin(x), 0, pi), (x * exp(x), 0, 1), (1 / (1 + x ** 2), -1, 1), (1 / x, 1, 7)]:
    A = integrate(f, x)
    D = integrate(f, x, a, b)
    print("int of", f, "from", a, "to", b, "=", D, "; antiderivative", A, "; F(b) - F(a) agrees:", bool((A.subs(x=b) - A.subs(x=a) - D).simplify_full() == 0))
