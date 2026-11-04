# Exercise exr-fam-sphere-moments (Maggiore, Sec. 6.1, eqs. (6.7), (6.8), (6.18), and Fig. 6.3): the surface charge
# densities s_d = s0 cos(theta) (odd under z -> -z) and s_Q = s0 sin(2 theta) cos(phi) = 2 s0 x z/R^2 (odd under
# x -> -x and under z -> -z) on the sphere of radius R. Charge q, dipole d_i, quadrupole Q_ij = int s (3 x_i x_j - r^2
# delta_ij) dA and the spherical moments q_lm = int s r^l conj(Y_lm) dA for l <= 2 are computed exactly.
th, ph, R, s0 = var('theta phi R s0'); assume(R > 0)
n = vector([sin(th)*cos(ph), sin(th)*sin(ph), cos(th)]); X = R*n
dA = R^2*sin(th)
def integ(f):
    return integrate(integrate((f*dA).simplify_trig(), ph, 0, 2*pi), th, 0, pi).simplify_full()
for name, s in (("s0 cos(theta)", s0*cos(th)), ("s0 sin(2 theta) cos(phi)", s0*sin(2*th)*cos(ph))):
    q = integ(s)
    d = [integ(s*X[i]) for i in range(3)]
    Q = matrix(SR, 3, 3, lambda i, j: integ(s*(3*X[i]*X[j] - R^2*(1 if i == j else 0))))
    print("density %s: q = %s, d = %s" % (name, q, d))
    print("  Q =", [list(r) for r in Q.rows()])
    qlm = {(l, m): integ(s*R^l*spherical_harmonic(l, m, th, ph).conjugate()) for l in range(3) for m in range(-l, l + 1)}
    print("  nonzero q_lm:", {k: v for k, v in qlm.items() if v != 0})
# the potential outside the second sphere is the single l = 2 term of prp-fam-multipole:
# Phi = (4 pi/5) sum_m q_2m Y_2m/r^3 = Q_ij n_i n_j/(2 r^3) = Q_xz sin(2 theta) cos(phi)/(2 r^3)
r = var('r')
Phi = sum(4*pi/5*integ(s0*sin(2*th)*cos(ph)*R^2*spherical_harmonic(2, m, th, ph).conjugate())*spherical_harmonic(2, m, th, ph) for m in (-1, 1))/r^3
print("Phi outside =", Phi.simplify_full().simplify_trig().factor())
