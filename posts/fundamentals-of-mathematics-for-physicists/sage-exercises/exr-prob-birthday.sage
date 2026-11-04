# Exercise exr-prob-birthday: the probability that two of n people share a birthday is 1 - prod_{k<n} (1 - k/365),
# computed exactly in QQ; it first exceeds 1/2 at n = 23.
def coincidence(n):
    return 1 - prod(1 - QQ(k) / 365 for k in range(n))
for n in [5, 10, 20, 22, 23, 30, 50]:
    print("n =", n, ": P(shared birthday) =", numerical_approx(coincidence(n), digits=4))
first = next(n for n in range(1, 100) if coincidence(n) > QQ(1) / 2)
print("smallest n with probability above 1/2:", first, "; its value", numerical_approx(coincidence(first), digits=6), "(an exact rational with 53-digit numerator and denominator)")
