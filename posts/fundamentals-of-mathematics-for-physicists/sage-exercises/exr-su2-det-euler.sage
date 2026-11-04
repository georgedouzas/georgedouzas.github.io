# Exercise exr-su2-det-euler: the Euler-angle matrix g = e^{phi T3} e^{theta T1} e^{psi T3}, with T_a = -(i/2) sigma_a,
# computed from the matrix exponentials: it has the SU(2) form with a = cos(theta/2) e^{-i(phi+psi)/2} and
# b = -i sin(theta/2) e^{i(phi-psi)/2}, hence det g = |a|^2 + |b|^2 = 1 and g is unitary.
th, ph, ps = var("th ph ps", domain="real")
s1 = matrix(SR, [[0, 1], [1, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
T1 = -I/2*s1; T3 = -I/2*s3
simp = lambda e: e.rectform().simplify_full().trig_reduce()      # half-angle mixtures need trig_reduce after simplify_full
clean = lambda M: M.apply_map(simp)
zero = lambda e: bool(simp(e) == 0)
e_ph = clean((ph*T3).exp()); e_th = clean((th*T1).exp()); e_ps = clean((ps*T3).exp())
print("e^{phi T3} ="); print(e_ph)
print("e^{theta T1} ="); print(e_th)
g = clean(e_ph*e_th*e_ps)
a = cos(th/2)*exp(-I*(ph + ps)/2); b = -I*sin(th/2)*exp(I*(ph - ps)/2)
print("g[0,0] = a = cos(theta/2) e^{-i(phi+psi)/2}:", zero(g[0, 0] - a))
print("g[1,0] = b = -i sin(theta/2) e^{i(phi-psi)/2}:", zero(g[1, 0] - b))
print("g[0,1] = -conj(b) and g[1,1] = conj(a):", zero(g[0, 1] + conjugate(b)) and zero(g[1, 1] - conjugate(a)))
print("|a|^2 + |b|^2 = 1:", zero(a*conjugate(a) + b*conjugate(b) - 1))
print("det g = 1:", zero(g.det() - 1))
print("g is unitary, g^dagger g = 1:", bool(clean(g.conjugate_transpose()*g - identity_matrix(SR, 2)) == 0))
print("at theta = 0 only phi + psi matters: g(0, phi, psi) =", clean(g.subs(th=0)))
