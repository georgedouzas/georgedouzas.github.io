# Exercise exr-tensor-components: tensors on a free module with SageManifolds' FiniteRankFreeModule: elementary tensors,
# components, the dimension of V (x) V, a contraction, and the change of components under a change of basis.
V = FiniteRankFreeModule(QQ, 3, name="V")
e = V.basis("e")
u = V([1, 2, 3], basis=e, name="u"); v = V([1, 0, -1], basis=e, name="v")
t = u * v
print("u (x) v is a tensor of type", t.tensor_type(), "with components u^i v^j:", t.comp(e)[:])
print("dimension of V (x) V:", V.tensor_module(2, 0).rank(), "; of V* (x) V:", V.tensor_module(1, 1).rank())
a = V.linear_form(name="a"); a[:] = [1, 0, -1]
print("a(u) = a_i u^i =", a(u), "; contracting a with the first slot of u (x) v gives the vector a(u) v =", a.contract(t).comp(e)[:])
f = V.basis("f", from_family=(e[0] + e[1], e[1], e[2]))
print("components of u in the basis f = (e0 + e1, e1, e2):", u.comp(f)[:], "; of u (x) v:", t.comp(f)[:])
T = V.tensor((1, 1), name="T"); T[:] = [[1, 2, 0], [0, 3, 1], [0, 0, 5]]
print("the trace (contraction of the two indices) of the type (1,1) tensor T is basis independent:", T.trace(), "=", sum(T.comp(f)[i, i] for i in range(3)))
