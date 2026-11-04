# Exercise exr-rw-ehrenfest (Foulaadvand, Sec. 10.2, eqs. (10.2)-(10.5)): N particles in two halves of a box; at each
# step a particle chosen at random changes side. The number n on the left is a Markov chain with
# P(n -> n - 1) = n/N, P(n -> n + 1) = (N - n)/N. From n = N: exact mean against N/2 + (N/2)(1 - 2/N)^t, the
# stationary distribution against Binomial(N, 1/2), and the relative fluctuation against 1/sqrt(N).
for N in [8, 64]:
    T = matrix(QQ, N + 1, N + 1)
    for n in range(N + 1):
        if n > 0: T[n, n - 1] = n/N
        if n < N: T[n, n + 1] = (N - n)/N
    p = vector(QQ, [0]*N + [1])
    ok = True
    for t in range(1, 41):
        p = p*T
        ok = ok and (sum(k*p[k] for k in range(N + 1)) == N/2 + N/2*(1 - 2/N)^t)
    print("N = %d: mean = N/2 + (N/2)(1 - 2/N)^t exactly for t = 1..40: %s" % (N, ok))
    pi_ = vector(QQ, [binomial(N, k)/2^N for k in range(N + 1)])
    print("        binomial distribution is stationary: %s; relative fluctuation sqrt(N/4)/(N/2) = %.4f" % (pi_*T == pi_, (1/sqrt(N)).n()))
    q = vector(QQ, [0]*N + [1])
    for t in range(4*N): q = q*T
    print("        n changes parity at every step (period 2): after 4N steps P(n even) = %.4f" % sum(q[k] for k in range(0, N + 1, 2)).n())
set_random_seed(1)
N, steps = 400, 20000
n, traj = N, []
for t in range(steps):
    n += -1 if random() < n/N else 1
    traj.append(n)
tail = traj[5000:]
m = sum(tail)/len(tail); s = sqrt(sum((x - m)^2 for x in tail)/len(tail))
print("one run, N = 400: n(200), n(400) = %d, %d (mean %.1f, %.1f); time average after t = 5000 %.1f, relative fluctuation %.4f (1/sqrt(N) = %.4f)" % (traj[199], traj[399], N/2 + N/2*(1 - 2/N)^200, N/2 + N/2*(1 - 2/N)^400, m, s/m, 1/sqrt(N.n())))
