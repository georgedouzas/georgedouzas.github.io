# Exercise exr-kkf-common-unit: the largest unit u/R such that all charges (n + alpha)/R of a spectrum with the given twists
# (together with periodic bosons) are integer multiples of it; u = 1/k with k the least common multiple of the denominators.
for twists in ([0, QQ(1) / 2], [0, QQ(1) / 3, QQ(2) / 3], [QQ(1) / 2, QQ(1) / 4], [0, QQ(2) / 5]):
    k = lcm([QQ(a).denominator() for a in twists] + [1])
    u = QQ(1) / k
    check = all(((n + a) / u) in ZZ for a in twists + [0] for n in range(-5, 6))
    larger = [QQ(1) / j for j in range(1, k) if all(((n + a) * j) in ZZ for a in twists for n in range(-5, 6))]
    print("twists", twists, ": u =", u, ", order of the cover", k, ", all charges multiples of u:", check, ", no larger unit works:", larger == [])
