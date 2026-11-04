# Exercise exr-dist-bump: derivatives of h(t) = e^{-1/t} are polynomials in 1/t times e^{-1/t}, with limit 0 at 0+.
t = var("t")
h = exp(-1 / t)
for n in range(1, 4):
    d = diff(h, t, n).simplify_full()
    print("h^(%d)(t) =" % n, d.factor(), "; limit at 0+:", limit(d, t=0, dir="+"))
print("the bump e^{-1/(1 - x^2)} and its first two derivatives at x -> 1-:")
x = var("x")
phi = exp(-1 / (1 - x ** 2))
print([limit(diff(phi, x, n), x=1, dir="-") for n in range(3)])
