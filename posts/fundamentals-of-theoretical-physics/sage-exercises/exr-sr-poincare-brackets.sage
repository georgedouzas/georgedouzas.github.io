# Exercise exr-sr-poincare-brackets: the brackets of the Poincare algebra with 5x5 matrices.
eta = diagonal_matrix(QQ, [1, -1, -1, -1])
def M(m, n):
    A = zero_matrix(QQ, 5)
    for a in range(4):
        for b in range(4):
            A[a, b] = eta[m, a]*(1 if n == b else 0) - eta[n, a]*(1 if m == b else 0)
    return A
def Pl(m):                       # P_mu: translation along e_mu
    A = zero_matrix(QQ, 5); A[m, 4] = 1; return A
Pu = lambda m: sum(eta[m, n]*Pl(n) for n in range(4))     # P^mu
br = lambda X, Y: X*Y - Y*X
okMP = all(br(M(m, n), Pu(r)) == eta[n, r]*Pu(m) - eta[m, r]*Pu(n) for m in range(4) for n in range(4) for r in range(4))
okPP = all(br(Pu(m), Pu(n)) == 0 for m in range(4) for n in range(4))
okMM = all(br(M(m, n), M(r, s)) == eta[n, r]*M(m, s) - eta[m, r]*M(n, s) - eta[n, s]*M(m, r) + eta[m, s]*M(n, r)
           for m in range(4) for n in range(4) for r in range(4) for s in range(4))
print("[M, P] relations:", okMP, "  [P, P] = 0:", okPP, "  Lorentz relations:", okMM)
print("exp(a^mu P_mu) for a = (1, 2, 3, 4):"); print((sum(c*Pl(m) for m, c in enumerate([1, 2, 3, 4]))).exp())
