# Exercise exr-lg-int-affine: the affine group of the line through every section of the chapter: the Lie algebra and
# its Killing form (degenerate), the exponential map (a bijection onto the group), the Maurer-Cartan form g^{-1} dg and
# its equation, and the left-invariant volume form.
aff = LieAlgebra(QQ, {("X", "Y"): {"Y": 1}}, names="X,Y")
Xg, Yg = aff.gens()
print("[X, Y] =", Xg.bracket(Yg), "; Killing form:", aff.killing_form_matrix().list(), "; semisimple:", aff.is_semisimple(), "; solvable:", aff.is_solvable())
x, y = var("x y", domain="real")
X = matrix(SR, [[1, 0], [0, 0]]); Y = matrix(SR, [[0, 1], [0, 0]])
E = (x * X + y * Y).exp().simplify_full()
print("exp(xX + yY) =", E.list())
a, b = var("a b", domain="real")
print("inverse of exp: given (a, b) with a > 0, x = log a, y = b log(a)/(a - 1):", bool(E.subs(x=log(a), y=b * log(a) / (a - 1)).simplify_full() == matrix(SR, [[a, b], [0, 1]])))
A = Manifold(2, "Aff"); C.<a, b> = A.chart("a:(0,+oo) b")
g = matrix(SR, [[a, b], [0, 1]])
print("g^-1 dg/da =", (g.inverse() * diff(g, a)).list(), "; g^-1 dg/db =", (g.inverse() * diff(g, b)).list())
sX = A.one_form(1 / a, 0); sY = A.one_form(0, 1 / a)
print("d sigma^X =", sX.exterior_derivative().display(), "; d sigma^Y + sigma^X ^ sigma^Y = 0:", sY.exterior_derivative() + sX.wedge(sY) == 0)
print("volume form sigma^X ^ sigma^Y =", sX.wedge(sY).display())
