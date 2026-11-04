# Exercise exr-sr-unknown-components (Blennow-Ohlsson, Problems 1.124-1.125): unknown field components from the two invariants.
var('al Bx Bz', domain='real')
E = vector([al, -al, 0]); B = vector([0, 0, 2*al]); Ep = vector([0, 0, 2*al]); Bp = vector([Bx, al, Bz])
print(solve([Ep*Bp == E*B, Ep*Ep - Bp*Bp == E*E - B*B], [Bx, Bz]))
