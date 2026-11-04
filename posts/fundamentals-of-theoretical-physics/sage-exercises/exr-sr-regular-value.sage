# Exercise exr-sr-regular-value: the explicit preimage X = (1/2) Lambda eta S of a symmetric matrix S under the
# derivative dF_Lambda(X) = X^T eta Lambda + Lambda^T eta X, for Lambda = B(0.5 x).
eta = diagonal_matrix([1, -1, -1, -1])
g = 2/sqrt(3); v = 1/2
Lam = matrix(SR, [[g, g*v, 0, 0], [g*v, g, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
S = matrix(SR, 4, 4, lambda m, n: m + n)
X = Lam*eta*S/2
print("dF(X) - S =", (X.transpose()*eta*Lam + Lam.transpose()*eta*X - S).simplify_full() == 0)
