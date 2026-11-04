# Exercise exr-clm-ct-action-angle: action-angle variables of the oscillator are canonical and give H = omega I.
th, I_ = var("theta I_", domain="real"); m, w = var("m omega", domain="positive")
assume(I_ > 0)
q = sqrt(2 * I_ / (m * w)) * sin(th); p = sqrt(2 * I_ * m * w) * cos(th)
print("{q, p}_(theta, I) =", (diff(q, th) * diff(p, I_) - diff(q, I_) * diff(p, th)).canonicalize_radical().simplify_full())
print("H(q(theta, I), p(theta, I)) =", (p**2 / (2 * m) + m * w**2 * q**2 / 2).simplify_full())
