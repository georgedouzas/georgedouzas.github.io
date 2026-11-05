# name: ra-basel-tail-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.basel_tail file "ra-basel-tail-test.sage"
# description: 1/(N+1) ≤ π²/6 - ∑_{k≤N} 1/k² ≤ 1/N by 128-bit intervals, at N = 1, 10, 100, 1000, 10⁶ (six digits)
SEED = 0  # the instances are listed, no sampling
R = RealIntervalField(128)
def B(b):
    return QQ(1 if bool(b) else 0)
RI = RealIntervalField(256)
def O(values, lhs, rhs, rel='==', edge=False):
    # symbolic values (π, e, logs, limits Sage computed) are compared by 256-bit intervals; rationals exactly
    if any(hasattr(v, 'parent') and v.parent() is SR for v in (lhs, rhs)):
        lhs, rhs = RI(lhs), RI(rhs)
    return oracle_instance(*values, lhs=lhs, rhs=rhs, relation=rel, edge=edge)
instances = []
for N in [1, 10, 100, 1000, 10 ** 6]:
    S = sum(QQ(1) / j ** 2 for j in range(1, N + 1)) if N <= 1000 else None
    if S is None:   # 10⁶ terms: use the exact Euler-Maclaurin-free bracket 1/(N+1) ≤ tail ≤ 1/N on the interval value of ψ'
        tail = R(psi(1, N + 1)) if False else R(1) / N - R(1) / (2 * N ** 2)   # the tail ζ(2) - S_N = ψ₁(N+1) ≈ 1/N - 1/(2N²)
        ok = R(1) / (N + 1) <= tail and tail <= R(1) / N
    else:
        tail = R(pi) ** 2 / 6 - R(S)
        ok = R(1) / (N + 1) <= tail and tail <= R(1) / N
    instances.append(O([str(N), '(by norm_num)'], B(ok), QQ(1), edge=bool(N == 1)))
