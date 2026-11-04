# Exercise exr-thd-compressibility: response functions of the ideal gas, kappa_T = -(1/V)(dV/dp)_T = 1/p and
# alpha = (1/V)(dV/dT)_p = 1/T, and the cyclic relation (dp/dT)_V (dT/dV)_p (dV/dp)_T = -1 for the ideal gas and van der Waals.
p, V, T, N, a, b = var("p V T N a b"); assume(p > 0, V > 0, T > 0, N > 0, a > 0, b > 0)
Vp = N * T / p
kappa = (-diff(Vp, p) / Vp).simplify_full(); alpha = (diff(Vp, T) / Vp).simplify_full()
print("kappa_T =", kappa, "  alpha =", alpha)
# cyclic relation from implicit differentiation of g(p, V, T) = 0
for name, g in [("ideal gas", p * V - N * T), ("van der Waals", (p + a * N ** 2 / V ** 2) * (V - N * b) - N * T)]:
    dpdT = -diff(g, T) / diff(g, p); dTdV = -diff(g, V) / diff(g, T); dVdp = -diff(g, p) / diff(g, V)
    print(name, ": (dp/dT)_V (dT/dV)_p (dV/dp)_T =", (dpdT * dTdV * dVdp).simplify_full())
