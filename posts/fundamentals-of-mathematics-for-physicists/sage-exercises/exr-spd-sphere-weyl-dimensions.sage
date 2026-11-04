# Exercise exr-spd-sphere-weyl-dimensions: the multiplicity 2^[n/2] binom(l + n - 1, l) of each sign of the Dirac
# eigenvalue l + n/2 on S^n equals the dimension of a Spin(n + 1) representation of highest weight (l + 1/2, 1/2, ...).
for n in range(2, 7):
    rank = (n + 1) // 2
    cartan = ("B%d" if (n + 1) % 2 else "D%d") % rank
    W = WeylCharacterRing(cartan)
    rep = lambda hw: W(W._space.from_vector_notation(hw))
    dims = []
    for l in range(5):
        hw = [l + 1/2] + [1/2] * (rank - 1)
        d1 = rep(hw).degree()
        if (n + 1) % 2 == 0:
            assert rep(hw[:-1] + [-hw[-1]]).degree() == d1      # the second representation, for the other sign
        dims.append(d1)
    expected = [2 ^ (n // 2) * binomial(l + n - 1, l) for l in range(5)]
    print("S^%d, Spin(%d) of type %s: dimensions %s, formula %s, equal: %s" % (n, n + 1, cartan, dims, expected, dims == expected))
