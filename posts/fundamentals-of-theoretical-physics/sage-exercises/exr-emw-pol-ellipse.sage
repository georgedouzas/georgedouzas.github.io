# Exercise exr-emw-pol-ellipse: axes and orientation of the polarisation ellipse for Ex = 1, Ey = 1/2, delta = pi/3.
t = var('t'); d = pi / 3
Ex = cos(t); Ey = cos(t - d) / 2
r2 = Ex^2 + Ey^2
vals = [find_local_maximum(r2, 0, pi), find_local_minimum(r2, 0, pi)]
print('semi-axes: %.4f and %.4f' % (sqrt(vals[0][0]), sqrt(vals[1][0])))
tm = vals[0][1]
print('angle of the major axis: %.2f degrees' % (atan2(Ey.subs(t=tm), Ex.subs(t=tm)) * 180 / pi).n())
print('formula: psi = arctan(2 Ex Ey cos(delta)/(Ex^2 - Ey^2))/2 = %.2f degrees' % (atan(2 * 1 * 0.5 * cos(d) / (1 - 0.25)) / 2 * 180 / pi).n())
