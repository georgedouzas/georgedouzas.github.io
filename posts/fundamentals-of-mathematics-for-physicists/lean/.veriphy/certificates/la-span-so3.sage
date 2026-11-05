# name: la-span-so3
# technique: a span equal to a kernel (independence recipe, KERNEL), adapted to matrices: so(3) is the kernel of
#   A ↦ Aᵀ + A on the 3 × 3 matrices (flattened to ℝ⁹), spanned by L₁, L₂, L₃
# goal: antisymMatrices 3 = Submodule.span ℝ (Set.range ![L1, L2, L3])
# description: Sage checks that the flattened L₁, L₂, L₃ lie in the kernel and have its dimension, and reads the
#   coordinates of a kernel vector off a non-singular minor (entries); Lean checks the generators entry by entry, and
#   a kernel element's combination by linarith from the antisymmetry equations.
LC = 'KaluzaKlein.LinearAlgebra.LinearCombinations.'
L = [matrix(QQ, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(QQ, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
     matrix(QQ, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]  # EDIT: the generators of the goal
n = 3


def q(v):
    v = QQ(v)
    return str(v) if v.denominator() == 1 else f'({v.numerator()} / {v.denominator()})'


flat = matrix(QQ, [m.list() for m in L])
# the antisymmetry equations A_ij + A_ji = 0 as a 9 × 9 system on the flattened entries
equations = matrix(QQ, [[(1 if (k == i * n + j) else 0) + (1 if (k == j * n + i) else 0) for k in range(n * n)]
                        for i in range(n) for j in range(n)])
assert equations * flat.transpose() == 0, 'a generator is not antisymmetric'
assert flat.rank() == equations.right_kernel().dimension(), 'the generators do not span the kernel'
rows, cols = nonzero_minor(flat.transpose(), flat.nrows())
inverse = flat.transpose().matrix_from_rows(rows).inverse()
entry = lambda k: f'X {k // n} {k % n}'
coefficients = [' + '.join(f'{q(inverse[i, r])} * {entry(rows[r])}' for r in range(len(rows)) if inverse[i, r] != 0)
                for i in range(flat.nrows())]
print('coordinates:', coefficients)
pairs = ', '.join(f'e {i} {j}' for i in range(n) for j in range(i, n))
names = ', '.join(LC + f'L{k}' for k in (1, 2, 3))
proof = tactic(
    'apply le_antisymm\n'
    '· intro X hX\n'
    '  have h : Xᵀ = -X := hX\n'
    '  have e : ∀ i j, X j i = -X i j := fun i j => by simpa using congrFun (congrFun h i) j\n'
    f'  refine (Submodule.mem_span_range_iff_exists_fun ℝ).2 ⟨![{", ".join(coefficients)}], ?_⟩\n'
    '  ext i j\n'
    f'  fin_cases i <;> fin_cases j <;> simp [Fin.sum_univ_succ, {names}] <;> linarith [{pairs}]\n'
    '· rw [Submodule.span_le]\n'
    '  rintro _ ⟨i, rfl⟩\n'
    '  fin_cases i <;>\n'
    f'    (show Matrix.transpose _ = -_; ext a b; fin_cases a <;> fin_cases b <;> simp [{names}])'
)
