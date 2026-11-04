# Exercise exr-clm-osc-damped-lagrangian: L = e^(2 gamma t)(q'^2 - omega^2 q^2)/2 gives the damped oscillator equation.
t, g, w, Q, V = var("t gamma omega Q V")
L = exp(2 * g * t) * (V**2 - w**2 * Q**2) / 2
q = function("q")(t)
on = {Q: q, V: diff(q, t)}
EL = diff(diff(L, V).subs(on), t) - diff(L, Q).subs(on)
print("EL / e^(2 gamma t) =", (EL / exp(2 * g * t)).simplify_full())
