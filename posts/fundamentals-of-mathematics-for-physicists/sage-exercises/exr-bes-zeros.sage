# Exercise exr-bes-zeros: the first three positive zeros of J_0 and J_1, their spacing, and their interlacing.
x = var("x")
def zeros(nu, brackets):
    return [find_root(bessel_J(nu, x), a, b) for a, b in brackets]
z0 = zeros(0, [(2, 3), (5, 6), (8, 9)])
z1 = zeros(1, [(3, 4.5), (6.5, 7.5), (9.5, 11)])
print("zeros of J_0:", [round(z, 4) for z in z0])
print("zeros of J_1:", [round(z, 4) for z in z1])
print("spacings of the J_0 zeros:", [round(z0[i + 1] - z0[i], 4) for i in range(2)], "compared with pi =", round(float(pi), 4))
print("interlacing j_{0,1} < j_{1,1} < j_{0,2} < j_{1,2} < j_{0,3} < j_{1,3}:", z0[0] < z1[0] < z0[1] < z1[1] < z0[2] < z1[2])
