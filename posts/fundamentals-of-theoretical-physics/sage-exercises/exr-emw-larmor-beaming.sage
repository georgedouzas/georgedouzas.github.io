# Exercise exr-emw-larmor-beaming: the direction of maximal emission for acceleration parallel to the velocity.
al, v = var('alpha v'); assume(v > 0, v < 1)
f = sin(al)^2 / (1 - v * cos(al))^5
c = (sqrt(1 + 15 * v^2) - 1) / (3 * v)
print('derivative at the stated cosine:', diff(f, al).subs(al=arccos(c)).simplify_full())
for vv in [0.5, 0.9, 0.99]:
    g = 1 / sqrt(1 - vv^2)
    print('v = %.2f: alpha_max = %.2f deg, 1/(2 gamma) = %.2f deg' % (vv, (arccos(c.subs(v=vv)) * 180 / pi).n(), (1 / (2 * g) * 180 / pi).n()))
