# Exercise exr-lt-dice-averages-sage: ten seeded runs of 10^4 die rolls; the root-mean-square deviation of the running
# averages from 7/2 at n = 10, 100, 1000, 10^4 against sigma/sqrt(n), sigma^2 = 35/12.
import random
set_random_seed(5); random.seed(int(5))
N = 10 ** 4
checkpoints = [10, 100, 1000, 10000]
averages = {n: [] for n in checkpoints}
for run in range(10):
    s = 0
    for i in range(1, N + 1):
        s += random.randint(1, 6)
        if i in averages:
            averages[i].append(QQ(s) / i)
sigma = sqrt(QQ(35) / 12)
for n in checkpoints:
    rms = sqrt(sum((a - QQ(7) / 2) ** 2 for a in averages[n]) / 10)
    print("n = %5d: averages %s" % (n, [float("%.3f" % float(a)) for a in averages[n]]))
    print("          rms deviation from 3.5 = %.4f, predicted sigma/sqrt(n) = %.4f" % (float(rms), float(sigma / sqrt(n))))
