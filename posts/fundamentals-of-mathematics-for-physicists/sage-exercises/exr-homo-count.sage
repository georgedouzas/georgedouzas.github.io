# Exercise exr-homo-count: all homomorphisms Z_m -> Z_n by brute force over the image a of the generator 1:
# k -> k a is well defined and a homomorphism exactly when m a = 0 in Z_n; their number is gcd(m, n).
def homs(m, n):
    out = []
    for a in range(n):
        f = {k: Zmod(n)(k * a) for k in range(m)}
        if all(f[(j + k) % m] == f[j] + f[k] for j in range(m) for k in range(m)):
            out.append(a)
    return out
for (m, n) in [(6, 3), (3, 6), (4, 6), (6, 4), (5, 7)]:
    h = homs(m, n)
    print("Hom(Z_%d, Z_%d): images of 1 = %s ; number %d = gcd(%d, %d) = %d" % (m, n, h, len(h), m, n, gcd(m, n)))
