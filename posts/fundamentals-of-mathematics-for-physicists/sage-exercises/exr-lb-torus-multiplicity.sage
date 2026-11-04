# Exercise exr-lb-torus-multiplicity: the multiplicities r_2(N) of the eigenvalues N = m^2 + n^2 <= 30 of the Laplacian on
# the square torus, the eigenvalues of multiplicity 8 and 12, and the missing integers.
from collections import Counter
cnt = Counter(m ** 2 + n ** 2 for m in range(-6, 7) for n in range(-6, 7))
table = [(N, cnt[N]) for N in range(31)]
print("N : r_2(N) for N <= 30:"); print(table)
print("multiplicity 8:", [N for N, r in table if r == 8], "; multiplicity 12:", [N for N, r in table if r == 12])
missing = [N for N, r in table if r == 0]
print("not sums of two squares:", missing)
print("each has a prime = 3 mod 4 to an odd power:", all(any(p % 4 == 3 and e % 2 == 1 for p, e in factor(N)) for N in missing if N > 0))
