# Exercise exr-csd-monopole-rotation: the monopole potential A = cos(th) dph on S^2 is symmetric under rotations:
# for each Killing vector xi_A of the round sphere, the Lie derivative L_xi A = d(i_xi A) + i_xi dA is exact,
# L_xi A = dW_A, so a rotation acts on A as an abelian gauge transformation with parameter W_A.
th, ph = var("th ph")
A = [0, cos(th)]                                   # components (A_th, A_ph)
F = diff(A[1], th) - diff(A[0], ph)                # dA = F dth ^ dph
xis = {"xi_1": [-sin(ph), -cot(th) * cos(ph)], "xi_2": [cos(ph), -cot(th) * sin(ph)], "xi_3": [0, 1]}
def lie(xi):
    iA = xi[0] * A[0] + xi[1] * A[1]
    iF = [-xi[1] * F, xi[0] * F]                   # i_xi (F dth ^ dph) = F (xi^th dph - xi^ph dth)
    return [(diff(iA, v) + iF[k]).simplify_full() for k, v in enumerate((th, ph))]
for name, xi in xis.items():
    L = lie(xi)
    W = integrate(L[1], ph).simplify_full()        # candidate potential from the ph component
    W = (W + integrate((L[0] - diff(W, th)).simplify_full(), th)).simplify_full()
    print(name, ": L_xi A =", L, ", W =", W)
    print("[PASS] %s: L_xi A = dW exactly:" % name, all(bool((L[k] - diff(W, v)).simplify_full() == 0) for k, v in enumerate((th, ph))))
