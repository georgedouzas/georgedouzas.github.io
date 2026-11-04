# Exercise exr-ems-dipole-axis-error: exact axial field of a loop over the dipole approximation.
u = var('u', domain='positive')   # u = z/a
ratio = (1 + 1 / u^2)^(-3/2)
print("ratio =", ratio, "; large-u expansion:", ratio.subs(u=1 / var('e')).taylor(var('e'), 0, 4))
print("ratio > 0.99 for z/a > %.3f" % find_root(ratio - 0.99, 1, 100))
for zz in [2, 3, 5, 10]:
    print("z = %d a: ratio %.4f" % (zz, ratio.subs(u=zz).n()))
