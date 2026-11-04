# Exercise exr-frob-legendre-second (Cini et al., Secs. 7.1.2-7.1.3): for Legendre's equation
# (1 - x^2) y'' - 2x y' + l(l + 1) y = 0, Abel's identity gives W = C/(1 - x^2); reduction of order
# y2 = y1 * integral(W/y1^2) gives the second solutions Q0 = artanh x and Q1 = x artanh x - 1, logarithmic at x = 1.
x = var("x")
for l, y1, y2 in [(0, 1, arctanh(x)), (1, x, x*arctanh(x) - 1)]:
    print("l = %d: (y2/y1)' - 1/((1 - x^2) y1^2) = %s | residual %s | Wronskian %s" % (l,
        (diff(y2/y1, x) - 1/((1 - x^2)*y1^2)).simplify_full(),
        ((1 - x^2)*diff(y2, x, 2) - 2*x*diff(y2, x) + l*(l + 1)*y2).simplify_full(),
        (y1*diff(y2, x) - diff(y1, x)*y2).simplify_full()))
print("against Sage's legendre_Q at x = 0.3:", [(legendre_Q(l, 0.3) - y2(x=0.3)).n() for l, y2 in [(0, arctanh(x)), (1, x*arctanh(x) - 1)]])
for X in [0.9, 0.99, 0.999]:
    print("x = %s: Q0 = %.4f, -log(1 - x)/2 + log(2)/2 = %.4f" % (X, arctanh(X), -log(1 - X)/2 + log(2.)/2))
