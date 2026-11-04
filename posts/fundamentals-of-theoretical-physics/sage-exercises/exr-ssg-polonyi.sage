# Exercise exr-ssg-polonyi: the Polonyi model, K = phi phibar, W = mu (phi + beta), units M_P = 1. Along real phi = x the
# potential V = e^K(|D W|^2 - 3|W|^2) is mu^2 e^{x^2}[(1 + x(x + beta))^2 - 3(x + beta)^2]. For beta = 2 - sqrt3 it has a
# minimum at x = sqrt3 - 1 with V = 0: supersymmetry is broken (D W != 0) with zero cosmological constant.
x, y, mu = var("x y mu", domain="real")
beta = 2 - sqrt(3)
phi = x + I * y
K = x^2 + y^2
W = mu * (phi + beta)
DW = mu + phi.conjugate() * W                         # D_phi W = W' + K_phi W with K_phi = phibar
V = (exp(K) * (DW * DW.conjugate() - 3 * W * W.conjugate())).simplify_full()
x0 = sqrt(3) - 1
grad = [diff(V, v).subs(x=x0, y=0).simplify_full() for v in (x, y)]
print("V(sqrt3 - 1, 0) =", V.subs(x=x0, y=0).simplify_full(), "   grad V there =", grad)
print("[PASS] phi = sqrt3 - 1 is a critical point with V = 0:", bool(V.subs(x=x0, y=0).simplify_full() == 0) and all(bool(g == 0) for g in grad))
H = matrix(SR, 2, 2, lambda i, j: diff(V, [x, y][i], [x, y][j]).subs(x=x0, y=0)).simplify_full()
ev = [e.simplify_full() for e in H.eigenvalues()]
print("Hessian eigenvalues / mu^2 =", [(e / mu^2).simplify_full().n(digits=6) for e in ev])
print("[PASS] both Hessian eigenvalues are positive, so it is a minimum:", all(bool((e / mu^2).n() > 0) for e in ev))
DW0 = DW.subs(x=x0, y=0).simplify_full(); m32 = (exp(K / 2) * W).subs(x=x0, y=0).simplify_full()   # W > 0 there for mu > 0
print("D W =", DW0, "  m_{3/2} = e^{K/2}|W| =", m32, "=", (m32 / mu).n(digits=6), "mu")
print("[PASS] D W = sqrt3 mu != 0: supersymmetry is broken, and |D W|^2 = 3|W|^2 makes V vanish:",
      bool((DW0 - sqrt(3) * mu).simplify_full() == 0) and bool((DW0^2 - 3 * W.subs(x=x0, y=0)^2).simplify_full() == 0))
