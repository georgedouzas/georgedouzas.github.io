# Exercise exr-grp-generators-zn: which elements of Z_12 generate it; the cyclic subgroup <k> has 12/gcd(k,12) elements.
R = Zmod(12)
def gen_sub(k):
    S = {R(0)}; x = R(k)
    while x not in S:
        S.add(x); x += R(k)
    return sorted(int(s) for s in S)
for k in range(12):
    print("<%2d> = %s" % (k, gen_sub(k)))
gens = [k for k in range(12) if len(gen_sub(k)) == 12]
print("generators of Z_12:", gens, "; these are the k with gcd(k,12) = 1:", gens == [k for k in range(12) if gcd(k, 12) == 1])
print("distinct cyclic subgroups:", sorted({tuple(gen_sub(k)) for k in range(12)}, key=len), "; one for each divisor of 12:", divisors(12))
