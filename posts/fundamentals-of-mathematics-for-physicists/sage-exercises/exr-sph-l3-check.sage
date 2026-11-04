# Exercise exr-sph-l3-check: orthonormality and ladder relations of the Y_3m, and r^3 Y_33 as a multiple of (x + iy)^3.
th, ph = var("theta phi")
Y = lambda m: spherical_harmonic(3, m, th, ph)
def inner(f, g):
    return integrate(integrate((f * conjugate(g)).simplify_full() * sin(th), ph, 0, 2 * pi), th, 0, pi).simplify_full()
ms = list(range(-3, 4))
G = matrix(SR, [[inner(Y(a), Y(b)) for b in ms] for a in ms])
print("the Gram matrix of Y_3m, m = -3..3, is the identity:", bool(G == identity_matrix(7)))
Lp = lambda f: exp(I * ph) * (diff(f, th) + I * cot(th) * diff(f, ph))
ok = all(bool((Lp(Y(m)) - sqrt((3 - m) * (4 + m)) * Y(m + 1)).simplify_full().canonicalize_radical() == 0) for m in range(-3, 3))
print("L_+ Y_3m = sqrt((3 - m)(4 + m)) Y_{3,m+1} for m = -3..2:", ok, "; L_+ Y_33 = 0:", bool(Lp(Y(3)).simplify_full() == 0))
x, y, z = var("x y z")
r = sqrt(x ** 2 + y ** 2 + z ** 2)
cart = (r ** 3 * Y(3)).subs({th: acos(z / r), ph: atan2(y, x)})
target = -sqrt(35 / (64 * pi)) * (x + I * y) ** 3
print("r^3 Y_33 = -sqrt(35/64pi) (x + iy)^3 at three points:", all(bool(abs((cart - target).subs({x: a, y: b, z: c}).n()) < 1e-10) for a, b, c in [(1, 2, 3), (0.5, -0.7, 2.1), (3, 1, -0.4)]))
