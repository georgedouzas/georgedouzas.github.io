# Exercise exr-cfb-aberration (Maggiore, Sec. 9.6): a source moving with velocity beta along x emits light of frequency w0
# in its rest frame at angle t' to x. Transforming the wave four-vector k = w (1, cos t, sin t, 0) gives the Doppler shift
# w = w0 / (gamma (1 - beta cos t)) and the aberration cos t = (cos t' + beta)/(1 + beta cos t'); half of an isotropic
# emission goes into the cone cos t > beta.
b, tp, w0 = var('b tp w0'); assume(0 < b, b < 1, w0 > 0)
g = 1 / sqrt(1 - b^2)
kp = vector([w0, w0 * cos(tp), w0 * sin(tp), 0])                   # in the source frame
Lb = matrix(SR, [[g, g * b, 0, 0], [g * b, g, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])   # source frame -> lab frame
k = Lb * kp
w = k[0]; ct = (k[1] / k[0]).simplify_full()
print("lab frequency w =", w.simplify_full(), "; null:", bool((k[0]^2 - k[1]^2 - k[2]^2).simplify_full() == 0))
print("cos t = (cos t' + b)/(1 + b cos t'):", bool((ct - (cos(tp) + b) / (1 + b * cos(tp))).simplify_full() == 0))
print("w = w0/(gamma (1 - b cos t)):", bool((w - w0 / (g * (1 - b * ct))).simplify_full() == 0))
print("emitted at t' = pi/2 -> cos t =", ct.subs(tp=pi / 2), ", sin t = 1/gamma:", bool((sqrt(1 - ct.subs(tp=pi / 2)^2) - 1 / g).simplify_full() == 0))
for bv in (0.5, 0.9, 0.99):
    print("beta = %.2f: half-angle of the forward cone %.2f deg, transverse Doppler w/w0 at t = 90 deg: %.4f" %
          (bv, (arccos(bv) * 180 / pi).n(), (1 / g).subs(b=bv).n()))
