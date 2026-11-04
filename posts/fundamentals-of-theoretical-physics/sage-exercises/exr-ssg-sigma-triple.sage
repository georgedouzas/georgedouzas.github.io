# Exercise exr-ssg-sigma-triple: the products of three and four sigma matrices in the signature (+,-,-,-), with
# sigma^mu = (1, sigma_i), sigmabar^mu = (1, -sigma_i) and epsilon_{0123} = 1 (so epsilon^{0123} = -1). Nath writes these
# identities in the mostly-plus metric; here the coefficients are found and checked for all index values.
K.<I> = QuadraticField(-1)
p = [matrix(K, [[0, 1], [1, 0]]), matrix(K, [[0, -I], [I, 0]]), matrix(K, [[1, 0], [0, -1]])]
one = identity_matrix(K, 2)
s = [one] + p; sb = [one] + [-x for x in p]
eta = diagonal_matrix(K, [1, -1, -1, -1])
def eps_dn(*ix):
    return Permutation([i + 1 for i in ix]).signature() if len(set(ix)) == 4 else 0
def eps_up(*ix):
    return -eps_dn(*ix)                       # raising four indices multiplies by det(eta) = -1
s_dn = [sum(eta[r, k] * s[k] for k in range(4)) for r in range(4)]       # sigma_rho
R4 = range(4)
def rhs3(m, n, l, c):
    return eta[m, n] * s[l] - eta[m, l] * s[n] + eta[n, l] * s[m] + c * I * sum(eps_up(m, n, l, r) * s_dn[r] for r in R4)
c3 = [c for c in (1, -1) if all(s[m] * sb[n] * s[l] == rhs3(m, n, l, c) for m in R4 for n in R4 for l in R4)]
print("sigma^mu sigmabar^nu sigma^lambda = eta^{mu nu} sigma^lambda - eta^{mu lambda} sigma^nu + eta^{nu lambda} sigma^mu + c i eps^{mu nu lambda rho} sigma_rho with c =", c3)
print("[PASS] the triple-product identity holds for all 64 index triples with a single sign c:", len(c3) == 1)
def rhs3b(m, n, l, c):
    return eta[m, n] * sb[l] - eta[m, l] * sb[n] + eta[n, l] * sb[m] + c * I * sum(eps_up(m, n, l, r) * sum(eta[r, k] * sb[k] for k in R4) for r in R4)
c3b = [c for c in (1, -1) if all(sb[m] * s[n] * sb[l] == rhs3b(m, n, l, c) for m in R4 for n in R4 for l in R4)]
print("[PASS] sigmabar sigma sigmabar obeys the same identity with the opposite sign of the epsilon term:", c3b == [-c3[0]])
def tr4(m, n, l, r):
    return (s[m] * sb[n] * s[l] * sb[r]).trace()
c4 = [c for c in (1, -1) if all(tr4(m, n, l, r) == 2 * (eta[m, n] * eta[l, r] - eta[m, l] * eta[n, r] + eta[m, r] * eta[n, l]) + c * 2 * I * eps_up(m, n, l, r)
                                 for m in R4 for n in R4 for l in R4 for r in R4)]
print("tr(sigma^mu sigmabar^nu sigma^lambda sigmabar^rho) = 2(eta eta - eta eta + eta eta) + c' 2i eps^{mu nu lambda rho} with c' =", c4)
print("[PASS] Nath's Problem 8.2 trace: tr[s^m sb^n s^l sb^r - s^l sb^n s^m sb^r] = c' 4i eps^{mnlr} for all 256 values:",
      all(tr4(m, n, l, r) - tr4(l, n, m, r) == c4[0] * 4 * I * eps_up(m, n, l, r) for m in R4 for n in R4 for l in R4 for r in R4))
