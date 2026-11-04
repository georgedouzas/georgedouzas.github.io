# Exercise exr-sr-inverse-compton (Blennow-Ohlsson, Problem 1.98): a photon scattered head on by a fast electron.
var('w wp th g m', domain='positive')
v = sqrt(1 - 1/g^2)
eta = diagonal_matrix([1, -1, -1, -1])
p = g*m*vector([1, -v, 0, 0]); k = w*vector([1, 1, 0, 0]); kp = wp*vector([1, cos(th), sin(th), 0])
sol = solve(p*eta*k - p*eta*kp - k*eta*kp == 0, wp)[0].rhs()      # (p + k - k')^2 = m^2 with k^2 = k'^2 = 0
claim = w*(1 + v)/(1 + v*cos(th) + w/(g*m)*(1 - cos(th)))
print("omega' minus the formula:", (sol - claim).simplify_full())
back = claim.subs(th=pi, g=1000, w=1, m=0.511e6)
print("backscattered energy for gamma = 1000, omega = 1 eV: %.4e eV (4 gamma^2 omega/(1 + 4 gamma omega/m) = %.4e)" % (n(back), 4e6/(1 + 4e3/0.511e6)))
