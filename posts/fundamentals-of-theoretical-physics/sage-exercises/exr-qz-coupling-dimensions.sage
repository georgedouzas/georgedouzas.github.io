# Exercise exr-qz-coupling-dimensions: mass dimensions of couplings in d dimensions, from [L] = d with
# [phi] = [A] = (d - 2)/2 and [psi] = (d - 1)/2.
d = var("d")
phi = (d - 2) / 2; psi = (d - 1) / 2; A = (d - 2) / 2
couplings = {"g3 phi^3": d - 3 * phi, "lambda phi^4": d - 4 * phi, "g6 phi^6": d - 6 * phi,
             "Yukawa g phi psibar psi": d - phi - 2 * psi, "gauge e psibar A psi": d - A - 2 * psi}
for k, v in couplings.items():
    print(k, ": dimension", v.simplify_full())
print("d   " + "  ".join(couplings.keys()))
for dv in [4, 5, 6, 10, 11]:
    print(dv, [v.subs(d=dv) for v in couplings.values()])
