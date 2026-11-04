# Exercise exr-ren-bs-1d-ising: the free energy per spin of the Ising chain from the inverse decimation map
# K = acosh(e^(2K'))/2, F(K) = log(2 sqrt(cosh 2K))/2 + F(K')/2, started from F(0.01) = log 2, against log(2 cosh K).
R = RealField(80)
Kp = R(0.01); F = log(R(2))
print("start: K = %.4f, F = %.10f, exact %.10f" % (Kp, F, log(2 * cosh(Kp))))
for step in range(1, 13):
    K = acosh(exp(2 * Kp)) / 2
    F = log(2 * sqrt(cosh(2 * K))) / 2 + F / 2
    Kp = K
    if step % 2 == 0:
        print("step %2d: K = %.5f, F = %.10f, exact %.10f, error %.1e" % (step, K, F, log(2 * cosh(K)), abs(F - log(2 * cosh(K)))))
