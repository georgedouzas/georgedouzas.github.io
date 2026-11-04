# Exercise exr-lgt-plaq-su3-trace: the trace of an SU(2) matrix is real, that of an SU(3) matrix is complex in general.
set_random_seed(5)
def haar_unitary(n):                         # QR of a complex Gaussian matrix, phases fixed, then det set to 1
    Z = random_matrix(CDF, n, n, algorithm='unitary')
    return Z / Z.det()^(1/n)
for n in (2, 3):
    U = haar_unitary(n)
    print('SU(%d): det = %s, Tr U = %s, Tr U^{-1} = %s' % (n, U.det().n(digits=6), U.trace().n(digits=6), U.inverse().trace().n(digits=6)))
U = haar_unitary(3)
print('SU(3): Tr U^{-1} = conjugate of Tr U:', abs(U.inverse().trace() - U.trace().conjugate()) < 1e-12)
