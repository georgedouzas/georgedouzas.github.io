# Exercise exr-cq-partition-count: Fock states of the modes omega_k = k (k >= 1) with total energy E are partitions
# of E; the generating function prod_k (1 - q^k)^{-1} is Tr q^H.
print("states of energy 4:", [list(p) for p in Partitions(4)])
print("p(E) for E = 0..10:", [Partitions(E).cardinality() for E in range(11)])
R.<q> = PowerSeriesRing(QQ, default_prec=11)
Z = prod(1 / (1 - q ** k) for k in range(1, 11))
print("prod_k (1 - q^k)^{-1} =", Z)
print("coefficients agree:", [Z[E] for E in range(11)] == [Partitions(E).cardinality() for E in range(11)])
