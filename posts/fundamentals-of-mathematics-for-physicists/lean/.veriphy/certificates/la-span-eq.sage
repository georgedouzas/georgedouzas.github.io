# name: la-span-eq
# technique: equality of two spans from the coordinates of every generator in the other family (independence recipe)
# goal: Submodule.span K (Set.range v) = Submodule.span K (Set.range w), both families written in the goal
# description: the families are read from the two sides of the goal; Sage finds the coordinates of each generator in
#   the other family; Lean checks both inclusions generator by generator.
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

left, right = re.split(r'\s=\s', goal.text, maxsplit=1)
VECTORS, OTHER = family(left), family(right)  # EDIT: read from the goal
print('left:', VECTORS, 'right:', OTHER)
proof = tactic(span_proof(VECTORS, other=OTHER, field=field(goal.text)))
