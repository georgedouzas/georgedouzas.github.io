# Exercise exr-vcr-torus-e7: the 70 scalars of eleven-dimensional supergravity on T^7 and the coset E7/SU(8).
# Metric moduli: dim GL(7)/SO(7) = 49 - 21 = 28; three-form C_mnp: binomial(7, 3) = 35; two-forms C_{mu nu m} in four
# dimensions, dual to scalars: 7. Compare with dim E7 - dim SU(8) computed from the root systems.
dimE7 = len(RootSystem(["E", 7]).root_lattice().roots()) + 7
dimSU8 = len(RootSystem(["A", 7]).root_lattice().roots()) + 7
moduli = 7 ^ 2 - binomial(7, 2)
print("dim E7 =", dimE7, "; dim SU(8) =", dimSU8, "; dim E7/SU(8) =", dimE7 - dimSU8)
print("metric moduli", moduli, "+ C_mnp", binomial(7, 3), "+ dualised C_mu nu m", 7, "=", moduli + binomial(7, 3) + 7)
print("vectors: 7 (metric) + 21 (C_mu mn) =", 7 + binomial(7, 2), "; with their 28 magnetic duals they fill the 56 of E7")
