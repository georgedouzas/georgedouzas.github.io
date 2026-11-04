# Exercise exr-vf-integral-k1: the integral curve of K1 = -sin(phi) d_theta - cot(theta) cos(phi) d_phi through the
# equator point (theta, phi) = (pi/2, pi/3): in R^3 it is the rotation about the X-axis, (1/2, sqrt3/2 cos t,
# sqrt3/2 sin t), and its chart coordinates solve the system theta' = -sin(phi), phi' = -cot(theta) cos(phi).
t = var("t")
th_t = arccos(sqrt(3)/2*sin(t)); ph_t = arctan(sqrt(3)*cos(t))
print("starting point (theta(0), phi(0)):", (th_t.subs(t=0), ph_t.subs(t=0)))
r1 = (diff(th_t, t) + sin(ph_t)).simplify_full()
r2 = (diff(ph_t, t) + cos(th_t)/sin(th_t)*cos(ph_t)).simplify_full()
print("residual of theta' = -sin(phi):", r1)
print("residual of phi' = -cot(theta) cos(phi):", r2)
print("residuals vanish at t = 0, pi/6, pi/4, pi/3:", all(bool(abs((ex).subs(t=s).n(digits=30)) < 1e-25) for ex in [r1, r2] for s in [0, pi/6, pi/4, pi/3]))
