# Exercise exr-dq-zitter-sage: for H = sigma_1 p + sigma_3 m, alpha(t) = e^{iHt} sigma_1 e^{-iHt} equals p H^-1 + e^{2iHt}(sigma_1 - p H^-1);
# the oscillation has frequency 2E and the time average is p H^-1.
p, m, t = var("p m t", domain="positive"); E = sqrt(p^2 + m^2)
s1 = matrix(SR, [[0, 1], [1, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]]); I2 = identity_matrix(SR, 2)
H = p * s1 + m * s3
ex = lambda c: cos(c * E * t) * I2 + I * sin(c * E * t) * H / E     # exp(i c H t), using H^2 = E^2
Hinv = H / E^2
alpha_t = ex(1) * s1 * ex(-1)
formula = p * Hinv + ex(2) * (s1 - p * Hinv)
print("H^2 = E^2:", (H * H - E^2 * I2).simplify_full() == 0)
print("alpha(t) = p H^-1 + e^{2iHt}(alpha - p H^-1):", (alpha_t - formula).simplify_full().simplify_trig() == 0)
avg = (alpha_t.apply_map(lambda e: integrate(e, t, 0, pi / E) * E / pi)).simplify_full()
print("time average over a period pi/E:", avg, "; equals p H^-1:", (avg - p * Hinv).simplify_full() == 0)
print("eigenvalues of p H^-1:", (p * Hinv).eigenvalues())
