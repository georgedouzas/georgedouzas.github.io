# Exercise exr-vcp-rs-conformal: for y > 0 the coordinate z = (e^{k y} - 1)/k turns the Randall-Sundrum metric
# e^{-2ky} eta dx dx - dy^2 into the conformally flat form (1 + k z)^{-2} (eta dx dx - dz^2): e^{-2ky} = (1 + kz)^{-2}
# and dy = dz/(1 + kz). The brane at y = 0 sits at z = 0 and y -> infinity is z -> infinity, the AdS horizon.
y, z, k = var("y z k", domain="positive")
zy = (exp(k * y) - 1) / k
yz = log(1 + k * z) / k
print("z(y(z)) = z:", bool(zy.subs(y=yz).simplify_full() == z))
print("e^{-2 k y} in terms of z:", exp(-2 * k * yz).simplify_full())
print("dy/dz =", diff(yz, z).simplify_full(), "; so dy^2 = dz^2 (1 + kz)^{-2}")
print("the conformal factor of dz^2 equals that of eta:", bool((diff(yz, z)**2 - exp(-2 * k * yz)).simplify_full() == 0))
