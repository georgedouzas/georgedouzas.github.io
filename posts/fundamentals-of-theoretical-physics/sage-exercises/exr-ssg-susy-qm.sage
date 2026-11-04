# Exercise exr-ssg-susy-qm: supersymmetric quantum mechanics with superpotential W(x) (hbar = 1, unit mass). With
# A = d/dx + W' and A^dag = -d/dx + W', the partner Hamiltonians are H_- = A^dag A/2 = (-d^2/dx^2 + W'^2 - W'')/2
# and H_+ = A A^dag/2 = (-d^2/dx^2 + W'^2 + W'')/2. For W = omega x^2/2 these are the oscillator shifted by -+ omega/2,
# and the zero mode of A is exp(-W), normalisable: unbroken supersymmetry. For W = g x^3/3 exp(-W) is not normalisable.
x, g = var("x g"); w = var("omega", domain="positive"); f = function("f")(x)
def A(W, u): return diff(u, x) + diff(W, x) * u
def Ad(W, u): return -diff(u, x) + diff(W, x) * u
for W in (w * x^2 / 2, g * x^3 / 3, x^4 / 4):
    Hm = (Ad(W, A(W, f)) / 2).expand(); Hp = (A(W, Ad(W, f)) / 2).expand()
    ok = bool((Hm - (-diff(f, x, 2) + diff(W, x)^2 * f - diff(W, x, 2) * f) / 2).expand() == 0) and \
         bool((Hp - (-diff(f, x, 2) + diff(W, x)^2 * f + diff(W, x, 2) * f) / 2).expand() == 0)
    print("[PASS] W = %s: A^dag A/2 = (-f'' + W'^2 f - W'' f)/2 and A A^dag/2 = (-f'' + W'^2 f + W'' f)/2:" % W, ok)
W = w * x^2 / 2
print("[PASS] W = omega x^2/2: H_-+ = (-d^2 + omega^2 x^2)/2 -+ omega/2, the bosonic and fermionic sectors of the oscillator:",
      bool((Ad(W, A(W, f)) / 2 - (-diff(f, x, 2) + w^2 * x^2 * f) / 2 + w / 2 * f).expand() == 0))
psi0 = exp(-W)
print("[PASS] A exp(-W) = 0, and exp(-omega x^2/2) is normalisable for omega > 0: int exp(-omega x^2) dx =",
      integrate(exp(-2 * W), x, -oo, oo), ":", bool(A(W, psi0) == 0))
W3 = g * x^3 / 3
print("[PASS] for W = g x^3/3 the zero modes exp(-W) and exp(+W) both blow up on one side: supersymmetry is broken:",
      bool(A(W3, exp(-W3)) == 0) and bool(limit(exp(-W3).subs(g=1), x=-oo) == oo) and bool(limit(exp(W3).subs(g=1), x=oo) == oo))
