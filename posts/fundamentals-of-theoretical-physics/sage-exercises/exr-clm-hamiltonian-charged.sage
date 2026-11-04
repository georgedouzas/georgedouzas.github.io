# Exercise exr-clm-hamiltonian-charged: Legendre transform of L = m v^2/2 - q phi + q v.A for arbitrary phi, A.
x, y, z, vx, vy, vz, px, py, pz = var("x y z vx vy vz px py pz", domain="real")
m, q = var("m q", domain="positive")
phi = function("phi")(x, y, z); A = vector([function("A%d" % i)(x, y, z) for i in (1, 2, 3)])
v = vector([vx, vy, vz]); P = vector([px, py, pz])
L = m / 2 * v * v - q * phi + q * v * A
mom = [diff(L, c) for c in v]
print("p = m v + q A:", all(bool(mom[i] == m * v[i] + q * A[i]) for i in range(3)))
sol = solve([P[i] == mom[i] for i in range(3)], [vx, vy, vz], solution_dict=True)[0]
H = (P * v - L).subs(sol)
print("H - (|p - qA|^2/(2m) + q phi) =", (H - ((P - q * A) * (P - q * A) / (2 * m) + q * phi)).simplify_full())
