# Exercise exr-rv-continuity-correction: exact P(|N - n/2| <= sqrt(n)/2) for Binomial(n, 1/2), n = 100 and 400,
# against the Gaussian erf(1/sqrt 2) and the continuity-corrected erf((h + 1/2)/sigma/sqrt 2) with h = sigma = sqrt(n)/2.
for n in [100, 400]:
    sigma = sqrt(QQ(n)) / 2
    pm = {j: binomial(n, j) / QQ(2) ** n for j in range(n + 1)}
    exact = sum(q for j, q in pm.items() if abs(j - QQ(n) / 2) <= sigma)
    crude = float(erf(1 / sqrt(2)))
    corrected = float(erf((sigma + QQ(1) / 2) / sigma / sqrt(2)))
    print("n = %d: exact %.5f, Gaussian %.5f (error %.4f), corrected %.5f (error %.4f)" % (n, float(exact), crude, float(exact) - crude, corrected, float(exact) - corrected))
