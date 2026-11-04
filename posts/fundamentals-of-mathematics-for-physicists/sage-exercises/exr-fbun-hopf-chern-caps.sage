# Exercise exr-fbun-hopf-chern-caps: Stokes' theorem on the two hemispheres. With A_+ = (cos(theta) - 1) d phi regular
# on the northern hemisphere and A_- = (cos(theta) + 1) d phi on the southern one, int_north F = oint A_+ (equator
# counterclockwise) and int_south F = -oint A_- ; the sum is -4 pi, the integral of F over the whole sphere.
th, ph, th0 = var("th ph th0", domain="real"); assume(th0 > 0, th0 < pi)
F = -sin(th)                                      # F = dA = -sin(theta) d theta ^ d phi
north = integrate(integrate(F, ph, 0, 2 * pi), th, 0, pi / 2)
south = integrate(integrate(F, ph, 0, 2 * pi), th, pi / 2, pi)
loop_p = integrate((cos(th) - 1).subs(th=pi / 2), ph, 0, 2 * pi)
loop_m = integrate((cos(th) + 1).subs(th=pi / 2), ph, 0, 2 * pi)
print("int_north F =", north, "; oint_equator A_+ =", loop_p)
print("int_south F =", south, "; -oint_equator A_- =", -loop_m)
print("total:", north + south, "= oint (A_+ - A_-) =", loop_p - loop_m, "; c_1 =", (north + south) / (2 * pi))
cap = integrate(integrate(F, ph, 0, 2 * pi), th, 0, th0)
print("flux through the cap theta < theta0:", cap.simplify_full(), "= oint A_+ on its boundary:",
      bool((cap - 2 * pi * (cos(th0) - 1)).simplify_full() == 0))
