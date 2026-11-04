# Exercise exr-fan-cosine-kernel: K f(x) = int_0^{2 pi} cos(x - y) f(y) dy on cos, sin, and functions orthogonal to them;
# eigenvalues of a discretisation.
x, y = var('x y')
K = lambda f: integrate(cos(x - y) * f.subs(x=y), y, 0, 2 * pi).simplify_full()
for f in (cos(x), sin(x), cos(2 * x), x - pi, SR(1)):
    print("K(%s) = %s" % (f, K(f)))
M = 200
h = 2 * pi.n() / M
A = matrix(RDF, M, M, lambda i, j: cos((i - j) * h) * h)
ev = sorted(A.eigenvalues(), key=lambda z: -abs(z))
print("largest eigenvalues of the discretised operator:", [round(abs(e), 6) for e in ev[:4]], " pi =", round(pi.n(), 6))
