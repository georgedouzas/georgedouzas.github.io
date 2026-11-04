# Exercise exr-exp-values: values of the complex exponential in Cartesian form, and its 2 pi i periodicity.
vals = {"e^(1 + i pi)": exp(1 + I*pi), "e^(i pi/2)": exp(I*pi/2), "e^(ln 2 + i pi/3)": exp(log(2) + I*pi/3), "e^(3 pi i/2)": exp(3*pi*I/2)}
for k, v in vals.items():
    print(k, "=", v.simplify_full())
z0 = 1 + I
print("e^(z + 2 pi i) == e^z for z = 1 + i (numerically):", abs(exp(z0 + 2*pi*I).n() - exp(z0).n()) < 1e-15)
