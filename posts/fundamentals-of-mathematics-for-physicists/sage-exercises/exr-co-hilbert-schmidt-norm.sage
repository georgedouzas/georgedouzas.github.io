# Exercise exr-co-hilbert-schmidt-norm: the Hilbert-Schmidt norm of the Green's kernel g(x, y) = min(x, y) - xy on [0, 1]^2,
# its agreement with sum 1/(n pi)^4, and the matrix elements of G in the sine basis.
x, y = var("x y", domain="real")
assume(x > 0, x < 1)
g_below = y - x * y      # g for y <= x (min = y)
g_above = x - x * y      # g for y >= x (min = x)
hs = integrate(integrate(g_below ** 2, y, 0, x) + integrate(g_above ** 2, y, x, 1), x, 0, 1)
n = var("n", domain="integer")
print("int int g^2 =", hs, "; sum 1/(n pi)^4 =", sum(1 / (n * pi) ** 4, n, 1, oo))
def elem(m, k):
    em = sqrt(2) * sin(m * pi * x); ek = sqrt(2) * sin(k * pi * y)
    inner = integrate(g_below * ek, y, 0, x) + integrate(g_above * ek, y, x, 1)
    return integrate(em * inner, x, 0, 1).simplify_full()
print("matrix elements <e_m, G e_k> for m, k <= 3:")
print(matrix(SR, 3, 3, lambda a, b: elem(a + 1, b + 1)))
print("diagonal = 1/(n pi)^2:", all(bool(elem(k, k) == 1 / (k * pi) ** 2) for k in [1, 2, 3]))
