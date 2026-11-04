# Exercise exr-sph-laplacian: the Laplacian on the sphere applied to cos^2 theta, sin theta cos phi and
# sin^2 theta cos 2 phi; the first is not an eigenfunction and splits into l = 0 and l = 2 parts.
th, ph = var("theta phi")
lap = lambda f: diff(sin(th) * diff(f, th), th) / sin(th) + diff(f, ph, 2) / sin(th) ** 2
for name, f, lam in [("cos^2 theta", cos(th) ** 2, None), ("sin theta cos phi", sin(th) * cos(ph), -2), ("sin^2 theta cos 2phi", sin(th) ** 2 * cos(2 * ph), -6)]:
    L = lap(f).simplify_full()
    if lam is None:
        print(name, ": Delta f =", L.trig_reduce(), "; not proportional to f")
    else:
        print(name, ": Delta f =", L, "; eigenvalue", lam, ":", bool((L - lam * f).simplify_full() == 0))
f0 = SR(1) / 3; f2 = cos(th) ** 2 - SR(1) / 3
print("cos^2 theta = 1/3 + (cos^2 theta - 1/3) with Delta(1/3) = 0 and Delta(cos^2 theta - 1/3) = -6 (cos^2 theta - 1/3):", bool(lap(f0) == 0) and bool((lap(f2) + 6 * f2).simplify_full() == 0))
