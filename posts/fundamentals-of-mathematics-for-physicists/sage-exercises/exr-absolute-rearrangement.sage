# Exercise exr-absolute-rearrangement: rearranging a conditionally convergent series changes its sum. Taking one
# positive term of the alternating harmonic series followed by two negative ones gives (1/2) log 2 instead of log 2.
def rearranged_partial_sum(blocks):
    total, p, q = QQ(0), 1, 2
    for _ in range(blocks):
        total += QQ(1) / p; p += 2                    # one odd (positive) term
        total -= QQ(1) / q; q += 2                    # two even (negative) terms
        total -= QQ(1) / q; q += 2
    return total
for b in [10, 100, 1000]:
    print("blocks =", b, ": rearranged partial sum =", numerical_approx(rearranged_partial_sum(b), digits=8))
print("(1/2) log 2 =", numerical_approx(log(2) / 2, digits=8), "; log 2 =", numerical_approx(log(2), digits=8))
print("an absolutely convergent series is immune: sum (-1)^(n+1)/n^2 = pi^2/12 =", numerical_approx(pi ** 2 / 12, digits=8), "whatever the order")
