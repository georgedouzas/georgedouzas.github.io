# name: set-topologies-count-test
# technique: test fallback (test_fallback recipe): kernel instances for n ≤ 2, Sage's exact counts for n = 3, 4, 5
# goal: sage_test KaluzaKlein.Sets.FiniteTopologies.topologies_count_claim (the numbers 1, 1, 4, 29, 355, 6942 of
#   topologies on n ≤ 5 points)
# description: the kernel evaluates the count for n = 0, 1, 2 by decide; Sage counts the families of subsets satisfying
#   the axioms directly for n = 3, 4 (2^8 and 2^16 families, the solution's enumeration), and for n = 5 counts the
#   systems of minimal open neighbourhoods U_x (x ∈ U_x, and y ∈ U_x implies U_y ⊆ U_x), which determine a topology on
#   a finite set; for n ≤ 4 both counts are computed and agree.
SEED = 0
TABLE = [1, 1, 4, 29, 355, 6942]

def count_families(n):
    full = (1 << n) - 1
    count = 0
    for fam in range(1 << (1 << n)):
        if not (fam & 1) or not (fam >> full) & 1:
            continue
        members = [s for s in range(1 << n) if (fam >> s) & 1]
        if all((fam >> (a | b)) & 1 and (fam >> (a & b)) & 1 for a in members for b in members):
            count += 1
    return count

def count_neighbourhoods(n):
    import itertools
    choices = [[U for U in range(1 << n) if (U >> x) & 1] for x in range(n)]
    count = 0
    for Us in itertools.product(*choices):
        if all(Us[y] & ~Us[x] == 0 for x in range(n) for y in range(n) if (Us[x] >> y) & 1):
            count += 1
    return count

instances = [kernel_instance(str(n), proof='decide', edge=(n == 0)) for n in (0, 1, 2)]
for n in (3, 4, 5):
    via = count_neighbourhoods(n)
    if n <= 4:
        direct = count_families(n)
        assert direct == via, (n, direct, via)
        print(f'Sage: n = {n}: {direct} families satisfy the axioms ({via} systems of minimal neighbourhoods)')
    else:
        print(f'Sage: n = {n}: {via} systems of minimal neighbourhoods')
    instances.append(oracle_instance(str(n), lhs=ZZ(via), rhs=ZZ(TABLE[n]), relation='=='))
