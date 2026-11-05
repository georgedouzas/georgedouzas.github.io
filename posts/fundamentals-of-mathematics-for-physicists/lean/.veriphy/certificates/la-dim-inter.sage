# name: la-dim-inter
# technique: the dimension of an intersection of two spans from a basis of it and the dimension formula (dimension
#   recipe, KIND = 'intersection')
# goal: Module.finrank K ↥(Submodule.span K (Set.range v) ⊓ Submodule.span K (Set.range w)) = d, v, w independent
#   families written in the goal
# description: Sage computes a basis of the intersection and an independent family spanning the sum; Lean checks
#   memberships by coordinates, independence by minors, and concludes by Veriphy.Linear.finrank_inf_of_bounds.
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

left, right = re.split(r'\s⊓\s', goal.text, maxsplit=1)
FIRST, SECOND = family(left), family(right)  # EDIT: read from the goal
print('first:', FIRST, 'second:', SECOND)
proof = tactic(intersection_dimension_proof(FIRST, SECOND, field(goal.text)))
