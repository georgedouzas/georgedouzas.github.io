# Exercise exr-spd-s2-eigenspinors: the lowest eigenspinors of the Dirac operator on the unit sphere, and the
# eigenspinors of eigenvalue 2 with phi-dependence e^{i phi/2} found by an ansatz in half-angle monomials.
th, ph, t = var("th ph t")
s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]])
def D(p):
    d_th = p.apply_map(lambda f: diff(f, th)); d_ph = p.apply_map(lambda f: diff(f, ph))
    return I * (s1 * (d_th + cot(th) / 2 * p) + s2 * d_ph / sin(th))
def is_zero(M):   # write theta = 2t and expand the trigonometric functions of 2t
    return all(bool((e * sin(th)).subs(th=2 * t).trig_expand().simplify_full() == 0) for e in M.list())
c, s = cos(th / 2), sin(th / 2)
for sg in (1, -1):
    psi = exp(I * ph / 2) * matrix(SR, 2, 1, [c, -sg * I * s])
    comp = exp(-I * ph / 2) * matrix(SR, 2, 1, [-sg * I * s, c])
    print("D psi = %+d psi for e^{i phi/2}(cos, %si sin):" % (sg, "-" if sg > 0 else "+"), is_zero(D(psi) - sg * psi),
          "; for e^{-i phi/2}(%si sin, cos):" % ("-" if sg > 0 else "+"), is_zero(D(comp) - sg * comp))
A = var(",".join("a%d" % k for k in range(4))); B = var(",".join("b%d" % k for k in range(4)))
mons = [c ^ k * s ^ (3 - k) for k in range(4)]
psi = exp(I * ph / 2) * matrix(SR, 2, 1, [sum(A[k] * mons[k] for k in range(4)), sum(B[k] * mons[k] for k in range(4))])
res = ((D(psi) - 2 * psi) / exp(I * ph / 2))
# the residual is linear in the 8 coefficients: sample it at several angles and find the null space numerically
import numpy as np
rows = []
for e in res.list():
    for tv in np.linspace(0.2, 2.9, 9):
        rows.append([complex(e.diff(x).subs(th=float(tv))) for x in list(A) + list(B)])
M = np.array(rows); sv = np.linalg.svd(M, compute_uv=False)
null = sum(1 for x in sv if x < 1e-9 * sv[0])
print("dimension of the solution space of D psi = 2 psi in the ansatz:", null)
v = np.linalg.svd(M)[2][-1].conj(); v = v / v[np.argmax(abs(v))]
print("solution (coefficients of cos^k sin^(3-k), k = 0..3, upper then lower):", [complex(round(z.real, 6), round(z.imag, 6)) for z in v])
