# Exercise exr-prg-massless-equal-time: the massless two-point function at equal times with a convergence factor
# e^{-eps k}: W(0, r) = (1/(4 pi^2 r)) int_0^oo sin(k r) e^{-eps k} dk = 1/(4 pi^2 (r^2 + eps^2)).
k, r, eps = var("k r eps"); assume(r > 0); assume(eps > 0)
W = 1 / (2 * pi) ** 3 * 4 * pi / (2 * r) * integrate(sin(k * r) * exp(-eps * k), k, 0, oo)
print("W(0, r) =", W.simplify_full())
print("equals 1/(4 pi^2 (r^2 + eps^2)):", bool(W == 1 / (4 * pi ** 2 * (r ** 2 + eps ** 2))))
print("limit eps -> 0:", limit(W, eps=0))
