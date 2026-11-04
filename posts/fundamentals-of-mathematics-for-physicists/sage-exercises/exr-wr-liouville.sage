# Exercise exr-wr-liouville (Cini et al., Sec. 7.1.2): the Wronskian of three solutions of y''' - 6y'' + 11y' - 6y = 0
# and Liouville's formula W' = (tr A) W for the companion system; then the Wronskian of three solutions of a
# second-order equation vanishes identically (at most two independent solutions).
t = var("t")
ys = [exp(t), exp(2*t), exp(3*t)]
W = matrix([[diff(y, t, k) for y in ys] for k in range(3)]).det().simplify_full()
A = companion_matrix([-6, 11, -6, 1], format="bottom")
print("solutions satisfy the equation:", all(bool((diff(y, t, 3) - 6*diff(y, t, 2) + 11*diff(y, t) - 6*y).simplify_full() == 0) for y in ys))
print("W(t) =", W, "| companion matrix A =", A.rows(), "| tr A =", A.trace())
print("W' - (tr A) W =", (diff(W, t) - A.trace()*W).simplify_full())
w = var("w")
zs = [cos(w*t), sin(w*t), cos(w*t + 1)]
print("three solutions of y'' + w^2 y = 0: 3x3 Wronskian =", matrix([[diff(z, t, k) for z in zs] for k in range(3)]).det().simplify_full())
