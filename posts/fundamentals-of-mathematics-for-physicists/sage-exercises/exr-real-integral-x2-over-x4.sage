# Exercise exr-real-integral-x2-over-x4: the integral of x^2/(1 + x^4) over the real line, by Sage's integrate and
# as 2 pi i times the sum of the residues in the upper half-plane.
x = var("x", domain="real"); z = var("z")
exact = integrate(x^2/(1 + x^4), x, -oo, oo)
print("integrate:", exact)
upper = [exp(I*pi/4), exp(3*I*pi/4)]
res = sum((z^2/(1 + z^4)).residue(z == w) for w in upper)
print("sum of upper residues:", res.simplify_full(), "  2 pi i * sum =", (2*pi*I*res).simplify_full())
print("equal:", bool((2*pi*I*res - exact).simplify_full() == 0))
