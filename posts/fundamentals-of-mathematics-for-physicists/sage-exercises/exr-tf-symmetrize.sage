# Exercise exr-tf-symmetrize: the symmetric and antisymmetric parts of a generic type (0, 2) tensor field on the plane
# with SageManifolds' symmetrize and antisymmetrize; they add up to the field, and their components have the stated
# symmetries.
M = Manifold(2, "R^2"); cart = M.chart("x y"); x, y = cart[:]
T = M.tensor_field(0, 2, name="T")
for i in range(2):
    for j in range(2):
        T[i, j] = function("T%d%d" % (i, j))(x, y)
Sym, Alt = T.symmetrize(), T.antisymmetrize()
print("symmetric part:", Sym.display()); print("antisymmetric part:", Alt.display())
print("T = Sym + Alt:", T == Sym + Alt)
print("Sym_xy = Sym_yx and Alt_xy = -Alt_yx:", bool(Sym[0, 1].expr() == Sym[1, 0].expr()), bool(Alt[0, 1].expr() == -Alt[1, 0].expr()))
dx, dy = cart.coframe()[0], cart.coframe()[1]
print("parts of dx (x) dy:", (dx*dy).symmetrize().display(), "and", (dx*dy).antisymmetrize().display())
