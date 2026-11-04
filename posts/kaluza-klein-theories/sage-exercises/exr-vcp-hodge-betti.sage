# Exercise exr-vcp-hodge-betti: Betti numbers of the quintic from its Hodge diamond, b_k = sum_{p+q=k} h^{p,q}, Poincare
# duality b_k = b_{6-k}, and the Euler number sum (-1)^k b_k = -200. In particular b_3 = 2 h^{2,1} + 2 = 204.
def diamond(h11, h21):
    h = {(p, q): 0 for p in range(4) for q in range(4)}
    for pq in [(0, 0), (3, 0), (0, 3), (3, 3)]: h[pq] = 1
    for pq in [(1, 1), (2, 2)]: h[pq] = h11
    for pq in [(2, 1), (1, 2)]: h[pq] = h21
    return h
for name, (h11, h21) in [("quintic", (1, 101)), ("mirror quintic", (101, 1)), ("T^6/Z_3 resolved", (36, 0))]:
    h = diamond(h11, h21)
    b = [sum(h[(p, k - p)] for p in range(4) if 0 <= k - p <= 3) for k in range(7)]
    chi = sum((-1) ** k * b[k] for k in range(7))
    print(name, ": Betti numbers", b, "; Poincare duality:", b == b[::-1], "; chi =", chi, "= 2(h11 - h21):", chi == 2 * (h11 - h21))
