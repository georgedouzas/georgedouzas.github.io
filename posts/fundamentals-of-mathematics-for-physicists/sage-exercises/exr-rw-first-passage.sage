# Exercise exr-rw-first-passage (Foulaadvand, Sec. 10.4.4, Fig. 10.25): the symmetric walk on 0..L absorbed at 0 and L.
# The mean number of steps tau(x) solves tau(x) = 1 + (tau(x - 1) + tau(x + 1))/2 with tau(0) = tau(L) = 0; the exact
# linear solve against x(L - x), then a seeded simulation of 2000 walks for L = 100.
L = 20
A = matrix(QQ, L - 1, L - 1); b = vector(QQ, [1]*(L - 1))
for i in range(L - 1):
    A[i, i] = 1
    if i > 0: A[i, i - 1] = -1/2
    if i < L - 2: A[i, i + 1] = -1/2
tau = A.solve_right(b)
print("L = 20: tau(x) = x(L - x) for all x:", all(tau[x - 1] == x*(L - x) for x in range(1, L)))
set_random_seed(2)
L = 100
for x0 in [10, 50]:
    total = 0; total2 = 0
    for trial in range(2000):
        x, n = x0, 0
        while 0 < x < L:
            x += 1 if randint(0, 1) else -1; n += 1
        total += n; total2 += n^2
    m = total/2000; se = sqrt((total2/2000 - m^2)/2000)
    print("L = 100, x0 = %d: simulated mean %.1f +- %.1f (standard error), x0(L - x0) = %d" % (x0, m, se, x0*(L - x0)))
