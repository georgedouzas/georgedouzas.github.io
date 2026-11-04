# Exercise exr-cs-charges-grassmannian: the Grassmannian of oriented 2-planes in R^4, SO(4)/SO(2)xSO(2), with
# r = span(M_01, M_23) and k = span(M_02, M_03, M_12, M_13): check that the decomposition is reductive and symmetric,
# and find the joint eigenvalues (the two isotropy charges) of ad_{M_01}, ad_{M_23} on k (x) C.
def M(i, j):
    m = matrix(QQ, 4, 4); m[i, j] = 1; m[j, i] = -1; return m
pairs = [(0,1), (2,3), (0,2), (0,3), (1,2), (1,3)]
B = [M(*p) for p in pairs]
coef = lambda X: vector(QQ, [X[p[0], p[1]] for p in pairs])
r, k = [0, 1], [2, 3, 4, 5]
br = lambda X, Y: X * Y - Y * X
print("[r, r] = 0:", all(br(B[i], B[j]) == 0 for i in r for j in r))
print("[r, k] in k (reductive):", all(all(coef(br(B[i], B[a]))[j] == 0 for j in r) for i in r for a in k))
print("[k, k] in r (symmetric):", all(all(coef(br(B[a], B[b]))[c] == 0 for c in k) for a in k for b in k))
ad = [matrix(QQ, 4, 4, lambda b, a: coef(br(B[i], B[k[a]]))[k[b]]) for i in r]
print("ad_{M_01} on k:", ad[0].rows()); print("ad_{M_23} on k:", ad[1].rows())
print("ad_{M_01} and ad_{M_23} commute:", ad[0] * ad[1] == ad[1] * ad[0])
K = QQbar
for v in (ad[0].change_ring(K) + 2 * ad[1].change_ring(K)).eigenvectors_right():   # a generic combination splits k (x) C
    vec = v[1][0]
    q = [ (ad[j].change_ring(K) * vec)[list(vec).index(next(x for x in vec if x != 0))] / next(x for x in vec if x != 0) for j in range(2)]
    print("joint eigenvalues (ad_{M_01}, ad_{M_23}) =", q, " charges", [x / I for x in q])
