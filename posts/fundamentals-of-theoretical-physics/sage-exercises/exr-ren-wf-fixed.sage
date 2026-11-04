# Exercise exr-ren-wf-fixed: the exact fixed points of the one-loop shell flow dr/dl = 2r + g/(1+r), dg/dl = eps g -
# 3 g^2/(1+r)^2 (n = 1, eps = 1), and the eigenvalues of the Jacobian.
var('r g')
fr = 2 * r + g / (1 + r); fg = 1 * g - 3 * g^2 / (1 + r)^2
sols = solve([fr == 0, fg == 0], r, g, solution_dict=True)
for s in sols:
    if s[g] != 0:
        J = matrix([[diff(fr, r), diff(fr, g)], [diff(fg, r), diff(fg, g)]]).subs(s)
        ev = sorted(J.eigenvalues(), key=lambda x: -x.n())
        print("fixed point r* = %s, g* = %s; eigenvalues %s; nu = %.4f" % (s[r], s[g], [e.n(digits=4) for e in ev], (1 / ev[0]).n()))
