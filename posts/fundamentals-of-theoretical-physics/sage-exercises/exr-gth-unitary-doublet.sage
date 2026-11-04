# Exercise exr-gth-unitary-doublet: every non-zero doublet phi = (phi1, phi2) is rotated by
# U = (1/|phi|) [[phi2, -phi1], [conj(phi1), conj(phi2)]] in SU(2) to (0, |phi|): the unitary gauge of the doublet.
a, b, c, d = var("a b c d", domain="real")
p1, p2 = a + I * b, c + I * d
N = sqrt(a^2 + b^2 + c^2 + d^2)
U = matrix(SR, [[p2, -p1], [p1.conjugate(), p2.conjugate()]]) / N
phi = vector(SR, [p1, p2])
simp = lambda M: M.apply_map(lambda e: e.simplify_full())
print("[PASS] U is unitary:", simp(U * U.conjugate_transpose()) == 1)
print("[PASS] det U = 1:", bool(U.det().simplify_full() == 1))
print("U phi =", (U * phi).apply_map(lambda e: e.simplify_full()))
print("[PASS] U phi = (0, |phi|):", bool((U * phi)[0].simplify_full() == 0) and bool(((U * phi)[1] - N).simplify_full() == 0))
num = {a: 1, b: -2, c: 3, d: 1}
print("example phi = (1 - 2i, 3 + i): U phi =", (U * phi).subs(num).simplify_full(), "= (0, sqrt(15))")
