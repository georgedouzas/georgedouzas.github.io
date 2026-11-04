# Exercise exr-clm-liou-damped: divergence of the damped oscillator flow and the determinant of exp(tA).
t = var("t", domain="real"); w, g = var("omega gamma", domain="positive")
A = matrix([[0, 1], [-w**2, -2 * g]])
print("divergence = trace A =", A.trace())
print("det exp(tA) =", (t * A).exp().det().simplify_full())
