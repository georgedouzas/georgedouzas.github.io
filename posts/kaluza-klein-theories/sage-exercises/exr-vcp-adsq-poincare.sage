# Exercise exr-vcp-adsq-poincare: Poincare coordinates on AdS_5 as the quadric
# (X^0)^2 + (X^5)^2 - (X^1)^2 - ... - (X^4)^2 = L^2 in the flat space of signature (+, -, -, -, -, +) (X^5 the second time).
# With x.x = t^2 - x1^2 - x2^2 - x3^2 and w > 0:
#   X^mu = L x^mu / w (mu = 0..3),  X^5 + X^4 = L^2 / w,  X^5 - X^4 = w - x.x / w,
# the point lies on the quadric and the induced metric is (L^2/w^2)(dt^2 - dx1^2 - dx2^2 - dx3^2 - dw^2).
# With L = 1/k and w = e^{k y}/k this is the Randall-Sundrum bulk e^{-2ky} eta dx dx - dy^2.
t, x1, x2, x3 = var("t x1 x2 x3"); w, L, k, y = var("w L k y", domain="positive")
x = [t, x1, x2, x3, w]
xx = t**2 - x1**2 - x2**2 - x3**2
Xp, Xm = L**2 / w, w - xx / w                      # X^5 + X^4 and X^5 - X^4
Xs = [L * t / w, L * x1 / w, L * x2 / w, L * x3 / w, (Xp - Xm) / 2, (Xp + Xm) / 2]
eta = diagonal_matrix(SR, [1, -1, -1, -1, -1, 1])
print("on the quadric:", bool((sum(eta[i, i] * Xs[i]**2 for i in range(6)) - L**2).simplify_full() == 0))
J = matrix(SR, 6, 5, lambda i, j: diff(Xs[i], x[j]))
g = (J.transpose() * eta * J).simplify_full()
print("induced metric = (L^2/w^2) diag(1, -1, -1, -1, -1):", (g - L**2 / w**2 * diagonal_matrix(SR, [1, -1, -1, -1, -1])).simplify_full() == 0)
# Randall-Sundrum form: w = e^{ky}/k, L = 1/k
conf = (L**2 / w**2).subs(L=1 / k, w=exp(k * y) / k).simplify_full()
dwdy = diff(exp(k * y) / k, y)
print("warp factor:", conf, "; coefficient of dy^2:", (conf * dwdy**2).simplify_full())
