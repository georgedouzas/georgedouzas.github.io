# Exercise exr-conformal-angles: tangent vectors of the images of two orthogonal curves through 1 + i under z^2 are
# orthogonal, each rotated by arg f'(z0) = pi/4 and scaled by |f'(z0)| = 2 sqrt(2).
t = var("t", domain="real")
z0 = 1 + I
g1, g2 = z0 + t, z0 + I*t
f = lambda z: z^2
T1 = diff(f(g1), t).subs(t=0); T2 = diff(f(g2), t).subs(t=0)
print("image tangents:", T1.expand(), ",", T2.expand())
v1, v2 = vector([T1.real(), T1.imag()]), vector([T2.real(), T2.imag()])
print("orthogonal (dot product 0):", v1*v2 == 0)
fp = 2*z0
print("f'(z0) =", fp, " |f'| =", abs(fp).simplify_full(), " arg f' =", arg(fp))
print("T1 == f'(z0) * 1 and T2 == f'(z0) * i:", bool(T1 == fp) and bool(T2 == fp*I))
