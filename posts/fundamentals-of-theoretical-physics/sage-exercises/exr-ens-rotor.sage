# Exercise exr-ens-rotor: the rigid rotor, E_l = l(l + 1)/(2I) with degeneracy 2l + 1; its partition function against
# the high-temperature value 2 I T (+ 1/3 + ...) and the low-temperature value 1 + 3 e^{-1/(I T)}.
def Zrot(t, L=400):          # t = 2 I T
    return sum((2 * l + 1) * exp(-l * (l + 1) / t) for l in range(L)).n()
for t in [0.2, 1, 10, 100]:
    print("2IT = %6s: Z = %.6f, 2IT + 1/3 = %.6f, 1 + 3 e^{-2/(2IT)} = %.6f" % (t, Zrot(t), t + 1 / 3, 1 + 3 * exp(-2 / t).n()))
l, t = var("l t"); assume(t > 0)
u = var("u"); assume(u > 0)
print("with u = l(l + 1), du = (2l + 1) dl: int_0^oo e^{-u/t} du =", integrate(exp(-u / t), u, 0, oo))
