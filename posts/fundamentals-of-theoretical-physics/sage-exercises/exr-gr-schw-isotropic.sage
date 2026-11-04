# Exercise exr-gr-schw-isotropic: with r = rho (1 + m/(2 rho))^2 the Schwarzschild metric becomes
# ((1 - m/2rho)/(1 + m/2rho))^2 dt^2 - (1 + m/2rho)^4 (drho^2 + rho^2 dOmega^2); to first order in m this is the weak-field
# metric g_tt = 1 + 2 Phi, g_ij = -(1 - 2 Phi) delta_ij with Phi = -m/rho.
m, rho, eps = var('m rho eps'); assume(rho > m/2, m > 0)
r = rho*(1 + m/(2*rho))^2
f = 1 - 2*m/r
gtt = f.simplify_full(); grr = (-(1/f)*diff(r, rho)^2).simplify_full(); gang = (-r^2/rho^2).simplify_full()
print("g_tt =", gtt.factor(), "; equals ((1 - m/2rho)/(1 + m/2rho))^2:", bool((gtt - ((1 - m/(2*rho))/(1 + m/(2*rho)))^2).simplify_full() == 0))
print("g_rho rho = g_angular/rho^2 = -(1 + m/2rho)^4:", bool((grr + (1 + m/(2*rho))^4).simplify_full() == 0), bool((gang + (1 + m/(2*rho))^4).simplify_full() == 0))
lin = lambda e: e.subs(m=eps*m).taylor(eps, 0, 1).subs(eps=1)
print("to first order: g_tt =", lin(gtt).expand(), "; g_rho rho =", lin(grr).expand())
