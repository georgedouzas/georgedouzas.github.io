# Exercise exr-complex-phase-rotation: the complex scalar Lagrangian is invariant under the global phase rotation
# phi -> e^{i alpha} phi but not under a local one alpha(x), whose failure is the term that the gauge field will cancel.
t, x, al = var("t x al", domain="real")
u, v, m = [function(n)(t, x) for n in ["u", "v"]] + [var("m", domain="real")]
phi = u + I * v; phib = u - I * v
L = (diff(phib, t) * diff(phi, t) - diff(phib, x) * diff(phi, x)) - m**2 * phib * phi          # eta^{mu nu} d_mu phibar d_nu phi, eta = diag(1, -1)
phi_r = exp(I * al) * phi; phib_r = exp(-I * al) * phib
L_r = (diff(phib_r, t) * diff(phi_r, t) - diff(phib_r, x) * diff(phi_r, x)) - m**2 * phib_r * phi_r
print("global rotation (alpha constant): L unchanged:", bool((L_r - L).simplify_full() == 0))
alf = function("alf")(t, x)
phi_l = exp(I * alf) * phi; phib_l = exp(-I * alf) * phib
L_l = (diff(phib_l, t) * diff(phi_l, t) - diff(phib_l, x) * diff(phi_l, x)) - m**2 * phib_l * phi_l
dL = (L_l - L).simplify_full()
print("local rotation (alpha(t, x)): L changes by terms proportional to d alpha:", bool(dL != 0), "; the change vanishes when d alpha = 0:", bool(dL.subs({diff(alf, t): 0, diff(alf, x): 0}).simplify_full() == 0))
print("|phi|^2 is invariant under any phase rotation:", bool((phib_l * phi_l - phib * phi).simplify_full() == 0))
