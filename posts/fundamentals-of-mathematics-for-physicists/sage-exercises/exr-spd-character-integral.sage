# Exercise exr-spd-character-integral: multiplicity of the U(1) charge q in V_j as (1/4pi) int_0^{4pi} e^{-i q a} chi_j(a) da.
a = var("a")
for twoj in range(0, 7):
    j = twoj / 2
    chi = sum(exp(I * (j - k) * a) for k in range(twoj + 1))
    row = []
    for q in (0, 1/2):
        val = numerical_integral(lambda t: real(exp(-I * q * t) * chi.subs(a=t)), 0, 4 * pi)[0] / (4 * pi.n())
        row.append("q = %s: %.6f" % (q, val))
    print("j = %s: " % j + ", ".join(row))
