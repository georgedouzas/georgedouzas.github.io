# Exercise exr-ren-forest-count: the number of forests of the s-channel chain of n bubbles; runs [i, j] of consecutive
# bubbles are the divergent subgraphs, compatible if nested or separated by at least one bubble.
import itertools
def forests(n):
    runs = [(i, j) for i in range(1, n + 1) for j in range(i, n + 1)]
    def ok(a, b):
        (i, j), (k, l) = sorted([a, b])
        return (i <= k and l <= j) or (k <= i and j <= l) or k >= j + 2
    return sum(1 for r in range(len(runs) + 1) for S in itertools.combinations(runs, r) if all(ok(a, b) for a, b in itertools.combinations(S, 2)))
counts = [forests(n) for n in range(1, 7)]
print("forests of the chain of n bubbles, n = 1..6:", counts)
schroeder = [sum(binomial(n + k, 2 * k) * catalan_number(k) for k in range(n + 1)) for n in range(1, 7)]
print("large Schroeder numbers S_1..S_6:              ", schroeder)
