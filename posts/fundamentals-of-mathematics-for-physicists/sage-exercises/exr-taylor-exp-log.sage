# Exercise exr-taylor-exp-log: Taylor series of e^z, Log(1 + z) and 1/(1 + z^2) about 0 to order 6, and their
# radii of convergence (infinite, 1 and 1).
z = var("z")
for f in (exp(z), log(1 + z), 1/(1 + z^2)):
    print("f =", f, ":  ", taylor(f, z, 0, 6))
print("1/(1 + z^2): coefficients (-1)^n z^{2n}, |a_n|^{1/n} -> 1, radius 1 (poles at +-i):", [r for r, m in (z^2 + 1).roots(ring=QQbar)])
print("e^z: ratio of coefficients a_{n+1}/a_n = 1/(n+1) -> 0, radius infinite:", [1/factorial(n + 1)*factorial(n) for n in range(5)])
