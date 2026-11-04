# Exercise exr-tensor-type-sage: tensors of several types on a free module of rank 3: their tensor_type() and their
# number of components 3^(r + s); the Kronecker delta as the type (1, 1) identity tensor.
V = FiniteRankFreeModule(QQ, 3, name="V")
e = V.basis("e")
for (r, s) in [(1, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]:
    t = V.tensor((r, s))
    print("type", t.tensor_type(), ": tensor module of rank", V.tensor_module(r, s).rank(), "= 3^", r + s)
delta = V.identity_map()
print("the identity map has type", delta.tensor_type(), "and components", delta.comp(e)[:])
