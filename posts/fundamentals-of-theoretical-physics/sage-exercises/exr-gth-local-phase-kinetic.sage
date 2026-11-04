# Exercise exr-gth-local-phase-kinetic: under a local phase rotation phi -> e^{i alpha(t,x)} phi the free complex
# scalar Lagrangian L = eta^{mu nu} d_mu phibar d_nu phi - m^2 phibar phi in 1+1 dimensions (signature (+, -))
# changes by -d_mu alpha j^mu + d_mu alpha d^mu alpha |phi|^2, with j^mu = i(phibar d^mu phi - phi d^mu phibar).
t, x, m = var("t x m", domain="real")
u, v, al = [function(n)(t, x) for n in ["u", "v", "al"]]
X = [t, x]; eta = [1, -1]
phi = u + I * v; phib = u - I * v
L = lambda f, fb: sum(eta[k] * diff(fb, X[k]) * diff(f, X[k]) for k in range(2)) - m^2 * fb * f
dL = (L(exp(I * al) * phi, exp(-I * al) * phib) - L(phi, phib)).simplify_full()
j = [I * eta[k] * (phib * diff(phi, X[k]) - phi * diff(phib, X[k])) for k in range(2)]       # j^mu, index raised
dal2 = sum(eta[k] * diff(al, X[k])^2 for k in range(2))                                       # d_mu alpha d^mu alpha
pred = -sum(diff(al, X[k]) * j[k] for k in range(2)) + dal2 * phib * phi
print("change of the Lagrangian under the local rotation:")
print(dL.expand())
print("[PASS] delta L = -d_mu alpha j^mu + (d_mu alpha d^mu alpha) |phi|^2:", bool((dL - pred).expand().simplify_full() == 0))
print("[PASS] the mass term m^2 |phi|^2 does not change (only derivatives see alpha(x)):", bool((exp(-I * al) * phib * exp(I * al) * phi - phib * phi).simplify_full() == 0))
