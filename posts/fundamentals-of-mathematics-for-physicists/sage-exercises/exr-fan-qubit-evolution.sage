# Exercise exr-fan-qubit-evolution: U = exp(-i theta n.sigma/2) by the spectral calculus, and U rho U^* for n = e3,
# rho = (1 + sigma_1)/2.
th = var('th'); assume(th, 'real')
s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
nv = vector(SR, [1, 2, 2]) / 3
ns = nv[0] * s1 + nv[1] * s2 + nv[2] * s3
U = (-I * th / 2 * ns).exp().simplify_full()
print("exp(-i th n.sigma/2) = cos(th/2) - i sin(th/2) n.sigma for n = (1,2,2)/3:",
      all(max(abs(z) for z in (U - (cos(th / 2) - I * sin(th / 2) * ns)).subs(th=v).change_ring(CDF).list()) < 1e-12 for v in (0.3, 1.7, 4.0)))
U3 = (-I * th / 2 * s3).exp()
rho = (1 + s1) / 2
r2 = (U3 * rho * U3.conjugate_transpose()).simplify_full()
bloch = [((r2 * s).trace()).simplify_full().simplify_trig() for s in (s1, s2, s3)]
print("Bloch vector of U rho U^*:", [b.rectform().simplify_full().simplify_trig() for b in bloch])
