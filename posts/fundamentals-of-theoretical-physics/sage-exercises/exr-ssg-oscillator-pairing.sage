# Exercise exr-ssg-oscillator-pairing: on the supersymmetric oscillator, with the basis |n, f> (n bosonic quanta,
# f = 0, 1 fermionic), Q = sqrt(omega) a^dag b maps |n - 1, 1> to sqrt(n omega) |n, 0> and annihilates every |n, 0>;
# Q^dag maps back, and Q^dag Q = H on the fermionic states, Q Q^dag = H on the bosonic ones.
N = 6; w = var("omega", domain="positive")
a = matrix(SR, N, N, lambda i, j: sqrt(j) if j == i + 1 else 0); b = matrix(SR, [[0, 1], [0, 0]])
A = a.tensor_product(identity_matrix(2)); Bf = identity_matrix(N).tensor_product(b)
H = w * (A.transpose() * A + Bf.transpose() * Bf); Q = sqrt(w) * A.transpose() * Bf; Qd = Q.transpose()
def ket(n, f):
    v = vector(SR, 2 * N); v[2 * n + f] = 1; return v
print("[PASS] Q |n-1, 1> = sqrt(n omega) |n, 0> for n = 1..5:", all(Q * ket(n - 1, 1) == sqrt(n * w) * ket(n, 0) for n in range(1, N)))
print("[PASS] Q |n, 0> = 0 and Q^dag |n, 1> = 0 for every n:", all((Q * ket(n, 0)).is_zero() and (Qd * ket(n, 1)).is_zero() for n in range(N)))
print("[PASS] Q^dag Q |n-1, 1> = H |n-1, 1> = n omega |n-1, 1>:", all(Qd * Q * ket(n - 1, 1) == n * w * ket(n - 1, 1) == H * ket(n - 1, 1) for n in range(1, N)))
print("[PASS] Q Q^dag |n, 0> = H |n, 0> = n omega |n, 0> below the truncation:", all(Q * Qd * ket(n, 0) == n * w * ket(n, 0) == H * ket(n, 0) for n in range(N)))
