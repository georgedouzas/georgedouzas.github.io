# Exercise exr-sph-coset-modes: the modes D^1_{m0}(phi, theta) on S^2 = SU(2)/U(1) from the spin-1 Wigner matrix, their
# orthogonality, and the expansion of cos theta + sin theta cos phi in them compared with the Y_1m expansion.
th, ph = var("theta phi")
d1 = matrix(SR, [[(1 + cos(th)) / 2, -sin(th) / sqrt(2), (1 - cos(th)) / 2], [sin(th) / sqrt(2), cos(th), -sin(th) / sqrt(2)], [(1 - cos(th)) / 2, sin(th) / sqrt(2), (1 + cos(th)) / 2]])
D = [exp(-I * m * ph) * d1[1 - m, 1] for m in [1, 0, -1]]
print("D^1_{m0} for m = 1, 0, -1:", D)
def inner(f, g):
    return integrate(integrate((f * conjugate(g)).simplify_full() * sin(th), ph, 0, 2 * pi), th, 0, pi).simplify_full()
G = matrix(SR, [[inner(a, b) for b in D] for a in D])
print("Gram matrix of the D^1_{m0} (expected 4 pi/3 times the identity):"); print(G)
f = cos(th) + sin(th) * cos(ph)
cD = [inner(f, Dm) / inner(Dm, Dm) for Dm in D]
print("coefficients of f in the D^1_{m0}:", [v.simplify_full() for v in cD])
Y = [spherical_harmonic(1, m, th, ph) for m in [1, 0, -1]]
cY = [inner(f, Ym) for Ym in Y]
print("coefficients of f in the Y_1m:", [v.simplify_full() for v in cY])
print("D^1_{m0} = sqrt(4 pi/3) conj(Y_1m):", all(bool((D[i] - sqrt(4 * pi / 3) * conjugate(Y[i])).simplify_full() == 0) for i in range(3)))
print("the two expansions agree: sum c_D D = sum c_Y Y = f:", bool((sum(cD[i] * D[i] for i in range(3)) - f).simplify_full() == 0) and bool((sum(cY[i] * Y[i] for i in range(3)) - f).simplify_full() == 0))
