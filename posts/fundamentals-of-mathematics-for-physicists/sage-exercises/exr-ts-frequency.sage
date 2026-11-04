# Exercise exr-ts-frequency (Foulaadvand, Secs. 3.2-3.3, eqs. (3.8)-(3.25), Figs. 3.2-3.5): the one-step matrices of
# the Euler, Euler-Cromer and velocity Verlet steps for x'' = -w^2 x, their determinants and traces, the quadratic
# form conserved by Euler-Cromer, and the frequency theta/tau of the discrete rotation against w.
w, tau, x, v = var("w tau x v")
E = matrix([[1, tau], [-w^2*tau, 1]])
C = matrix([[1 - w^2*tau^2, tau], [-w^2*tau, 1]])
xn = x + tau*v - w^2*tau^2*x/2
V = jacobian([xn, v - w^2*tau*(x + xn)/2], [x, v])
for name, M in [("Euler", E), ("Euler-Cromer", C), ("velocity Verlet", V)]:
    print("%-16s det = %s, trace = %s" % (name, M.det().simplify_full(), M.trace().simplify_full()))
X = C*vector([x, v]); Q = v^2 + w^2*x^2 - w^2*tau*x*v
print("Euler-Cromer conserves Q = v^2 + w^2 x^2 - w^2 tau x v:", bool((Q.subs(x=X[0], v=X[1]) - Q).expand() == 0))
Y = E*vector([x, v])
print("Euler multiplies v^2 + w^2 x^2 by", ((Y[1]^2 + w^2*Y[0]^2)/(v^2 + w^2*x^2)).simplify_full())
s = var("s")
print("theta/tau = (2/tau) arcsin(w tau/2) =", taylor(2*arcsin(w*s/2)/s, s, 0, 4))
print("w = 3: tau = 0.001, 0.005, 0.01 -> numerical frequency", [round((2/t*arcsin(3*t/2)).n(), 6) for t in (0.001, 0.005, 0.01)],
      "| Euler amplitude factor after t = 20:", [round(((1 + 9*t^2)^(10/t)).n(), 3) for t in (0.001, 0.005, 0.01)])
