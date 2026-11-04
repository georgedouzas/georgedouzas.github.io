# Exercise exr-ssg-susy-sqed-fi (integrative): supersymmetric QED with charges +1, -1, a Fayet-Iliopoulos term kappa > 0 and
# no superpotential: V = (g^2/2)(kappa/g - |phi_+|^2 + |phi_-|^2)^2 written as (1/2)(kappa - g(|phi_+|^2 - |phi_-|^2))^2. The
# vacua |phi_+|^2 - |phi_-|^2 = kappa/g are supersymmetric (V = 0) but break the gauge symmetry; at phi_- = 0 the photon gets
# mass^2 2 g^2 |phi_+|^2 = 2 g kappa and the radial scalar the same mass: together with the fermions a massive vector multiplet,
# 4 + 4 states. Adding W = m Phi_+ Phi_- makes V > 0 everywhere: supersymmetry is broken.
g, k = var("g kappa", domain="positive"); m = var("m", domain="positive")
r, s = var("r s", domain="positive")
V0 = (k - g * (r - s))^2 / 2
print("[PASS] without W the vacua are the D-flat surface r - s = kappa/g, with V = 0:", bool(V0.subs(r=s + k / g) == 0))
h = var("h", domain="positive")                                         # phi_+ = v + h/sqrt(2), real radial fluctuation
v = sqrt(k / g)
Vh = V0.subs(r=(v + h / sqrt(2))^2, s=0).expand()
mh2 = diff(Vh, h, 2).subs(h=0)
mA2 = 2 * g^2 * v^2                       # from |D_mu phi|^2 = ... + g^2 A^2 |phi|^2 with |phi|^2 = v^2: m_A^2 = 2 g^2 v^2
print("radial scalar mass^2 =", mh2.simplify_full(), "; vector mass^2 2 g^2 |phi_+|^2 =", mA2)
print("[PASS] the radial scalar and the photon have the same mass^2 2 g kappa: a massive vector multiplet:", bool(mh2.simplify_full() == 2 * g * k) and bool(mA2 == 2 * g * k))
print("[PASS] states: vector 3 + radial scalar 1 = 4 bosons, two Weyl fermions (gaugino, psi_+) married into a Dirac pair = 4 fermions:", 3 + 1 == 2 * 2)
V = m^2 * (r + s) + V0
print("[PASS] with W = m Phi_+ Phi_- V > 0 everywhere: at V = 0 both m^2(r + s) = 0 and r - s = kappa/g would be needed:",
      bool(V.subs(r=0, s=0) == k^2 / 2) and bool(V0.subs(r=0, s=0) > 0))
