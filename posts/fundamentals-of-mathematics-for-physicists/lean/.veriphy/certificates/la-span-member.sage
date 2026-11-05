# name: la-span-member
# technique: membership in a span from the coordinates of the vector (independence recipe, span from coordinates)
# goal: x ∈ Submodule.span K (Set.range ![![…], …]), the vector and the family written in the goal
# description: Sage solves for the coordinates of x in the family; Lean checks the combination entry by entry
#   (Submodule.mem_span_range_iff_exists_fun), as the recipe's span emitter does generator by generator.
import re


def _rational(x):
    return QQ(sage_eval(re.sub(r"[()]", "", x).strip()))


def family(text):
    """The explicit family ![![…], …] (or vector ![…]) written in the goal, as lists of rationals."""
    t = re.sub(r"\(([^():]*?) : [^()]*\)", r"\1", text)  # drop type ascriptions
    return [[_rational(x) for x in v.split(",")] for v in re.findall(r"!\[([^\[\]]*)\]", t)]


def literal(text):
    """The first matrix literal !![a, b; c, d] written in the goal, as a list of rows."""
    t = re.sub(r"\(([^():]*?) : [^()]*\)", r"\1", text)
    body = re.search(r"!!\[([^\[\]]*)\]", t).group(1)
    return [[_rational(x) for x in row.split(",")] for row in body.split(";")]


def field(text):
    return "ℝ" if "ℝ" in text else "ℚ"

vectors = family(goal.text)
TARGET, BASIS = vectors[0], vectors[1:]  # EDIT: read from the goal
K = field(goal.text)
c = span_coordinates(TARGET, BASIS)
print('coordinates:', c)
check = 'by ext j <;> fin_cases j <;> simp [Fin.sum_univ_succ, Pi.smul_apply, smul_eq_mul] <;> norm_num'
proof = tactic(f'exact (Submodule.mem_span_range_iff_exists_fun {K}).2 ⟨{lean_vector(c)}, {check}⟩')
