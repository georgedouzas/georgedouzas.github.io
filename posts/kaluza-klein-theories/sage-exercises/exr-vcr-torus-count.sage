# Exercise exr-vcr-torus-count: on-shell degrees of freedom of gravity on a torus. A graviton in D dimensions has
# D(D-3)/2 polarisations; after reduction on T^n to d = D - n dimensions it becomes a graviton (d(d-3)/2), n vectors
# (d - 2 each) and n(n+1)/2 scalars. Check the count symbolically and in the cases D = 5, 10, 11 with d = 4.
d, n = var("d n")
D = d + n
lhs = D * (D - 3) / 2
rhs = d * (d - 3) / 2 + n * (d - 2) + n * (n + 1) / 2
print("D(D-3)/2 - [d(d-3)/2 + n(d-2) + n(n+1)/2] =", (lhs - rhs).expand())
for DD in [5, 10, 11]:
    nn = DD - 4
    print("D = %d on T^%d: %d = 2 (graviton) + %d (vectors) + %d (scalars) = %d" % (DD, nn, DD * (DD - 3) // 2, 2 * nn, nn * (nn + 1) // 2, 2 + 2 * nn + nn * (nn + 1) // 2))
