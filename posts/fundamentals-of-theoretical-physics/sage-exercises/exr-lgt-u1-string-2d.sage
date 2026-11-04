# Exercise exr-lgt-u1-string-2d: the Polyakov-loop correlator of 2D compact U(1) on the L x L torus, exactly, and the string tension.
beta, L = 3, 10
I = lambda n: bessel_I(abs(n), beta)
Z = sum(I(n)^(L*L) for n in range(-8, 9))
def corr(R):                                 # area R L carries charge n + 1, the rest n
    return sum(I(n)^(L*L - R*L) * I(n + 1)^(R*L) for n in range(-8, 9)) / Z
u = I(1) / I(0); sig = -log(u)
for R in range(1, 6):
    print('R = %d: exact %.6e, u^{RL} + u^{(L-R)L} = %.6e' % (R, corr(R), u^(R*L) + u^((L - R)*L)))
print('slope -log(C(2)/C(1)) / L = %.5f, string tension log(I_0/I_1) = %.5f' % (-log(corr(2) / corr(1)) / L, sig))
