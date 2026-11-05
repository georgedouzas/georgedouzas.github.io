"""Check the per-chapter verification manifests (design D28-D30): every block of a chapter has its claims verified.

A manifest `verification/<chapter>.yaml` lists every `def-`/`thm-`/`prp-`/`lem-`/`cor-`/`exm-`/`exr-` block of
`tutorial/<chapter>.qmd`:

    chapter: part-i-02-linear-algebra
    blocks:
      - label: exm-basis-f
        claims:
          - claim: '(5, 4) = 2 f1 + f2'                  # the claim in words
            lean: KaluzaKlein.Linear.coordinates_f        # fully qualified Lean theorem (or definition, for a def- block)
            recipe: witness                               # a Veriphy recipe, or `mathlib`
            certificate: witness.sage                     # .veriphy/certificates/<file>, required for a Veriphy recipe
            why: '...'                                    # required when recipe is `mathlib`: why no recipe applies
      - label: thm-rank-nullity
        claims:
          - claim: 'rank + nullity = n for every matrix'
            lean: KaluzaKlein.Linear.rank_nullity
            recipe: mathlib                               # proved with mathlib; no certificate is feasible
            why: '...'
            test: rank_nullity                            # Veriphy's sage_test fallback: .veriphy/tests/<test>.json
            level: kernel-test                            # kernel-test or oracle-test, as the test record states
            instances: 5
      - label: exm-some-remark
        none: 'motivation only, states no value or identity'   # a block with no checkable claim, and why

Levels (design D29): a claim proved by a Veriphy recipe is at level `certificate`; a claim with a `test` is at the
level of its test (`kernel-test` or `oracle-test`, at least three instances, its record committed under
`.veriphy/tests/`); any other claim is `pending`. `--strict` (the gate) rejects pending claims. A block with claims has
a Verification callout; every statement has one (change kk-csdr-verification, design V4: definitions are verified
against their Lean counterparts, so `none` entries for definitions are to be replaced by correspondence claims).

Rules: every block of the chapter has exactly one entry and every entry is a block; a block has `claims` or `none`;
every claim names an existing Lean declaration, a known recipe (or `mathlib` with `why`), and for a Veriphy recipe a
committed certificate script; every Lean name of a block is shown in the block's Verification callout (the callout
after the block, or after the solution of an exercise). The axiom audit and `sorry` check stay in lean_audit.py.

Usage: python3 check_verification.py [chapter ...] [--strict]   (default: every manifest present); `--coverage` prints,
for every chapter, how many blocks are covered by a manifest; `--levels` prints the number of claims at each level.
"""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fold_blocks  # the reading wrappers of change kk-csdr-presentation are ignored
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lean_audit
import yaml

ROOT = os.path.normpath(os.path.join(HERE, ".."))
TUT = ROOT
MAN = os.path.join(ROOT, "verification")
CERTS = os.path.join(ROOT, "lean", ".veriphy", "certificates")
RECIPES = {"calc_derivation", "case_split", "certified_bound", "coordinate_geometry", "derivative", "factor_rewrite",
           "farkas", "fusion_category", "hint_search", "ideal_membership", "infeasibility", "lie_structure",
           "lie_superalgebra", "matrix_exponential", "matrix_from_goal", "matrix_identity", "matrix_normal_form",
           "ode_solution", "operator_identity", "primality", "rational_inequality", "recurrence", "representation",
           "root_isolation", "sos", "sum_induction", "transcendental_rewrite", "witness", "wz_certificate",
           "physlib/lorentz_algebra", "physlib/ladder_operators"}
BLOCK = re.compile(r"^:::+ \{#((?:def|thm|prp|lem|cor|exm|exr)-[\w-]+)[^}]*\}", re.M)
VERIF = re.compile(r"^:::+ \{\.callout-tip[^}]*\}\n(.*?)^:::+[ \t]*$", re.M | re.S)
LEANSC = re.compile(r"\{\{<\s*lean\s+([\w.']+)\s*>\}\}")
TESTS = os.path.join(ROOT, "lean", ".veriphy", "tests")
LEVELS = ("certificate", "kernel-test", "oracle-test", "pending")
EXTRA = os.path.join(ROOT, "verification", "recipes_extra.txt")  # recipes added on the Veriphy branch, one per line


def recipes():
    extra = set()
    if os.path.exists(EXTRA):
        extra = {l.strip() for l in open(EXTRA) if l.strip() and not l.startswith("#")}
    return RECIPES | extra


def declarations():
    names = set(lean_audit.theorems())
    for f in lean_audit.sources():  # definitions too (a def- block may be backed by a Lean definition)
        ns = []
        for line in lean_audit._strip_comments(open(f).read()).split("\n"):
            m = re.match(r"(namespace|end) (\S+)\s*$", line)
            if m:
                if m.group(1) == "namespace": ns.append(m.group(2))
                elif ns and ns[-1] == m.group(2): ns.pop()
                continue
            m = re.match(r"(?:@\[[^\]]*\]\s*)?(?:private |protected |noncomputable )*(?:def|abbrev|structure|instance|class) (\S+)", line)
            if m:
                names.add(".".join(ns + [m.group(1)]))
    return names


def shown(text):
    """Lean names shown in the Verification callout that follows each block (after the solution for an exercise)."""
    out, blocks = {}, [(m.group(1), m.start()) for m in BLOCK.finditer(text)]
    for i, (label, pos) in enumerate(blocks):
        end = blocks[i + 1][1] if i + 1 < len(blocks) else len(text)
        out[label] = set()
        for v in VERIF.finditer(text, pos, end):
            out[label] |= set(LEANSC.findall(v.group(1)))
    return out


def level(claim, known):
    """The verification level of one manifest claim (design D29)."""
    if claim.get("recipe") in known and claim.get("certificate"):
        return "certificate"
    if claim.get("test"):
        return claim.get("level", "pending")
    return "pending"


def has_callout(text):
    """Whether each block is followed by a Verification callout (after the solution, for an exercise)."""
    out, blocks = {}, [(m.group(1), m.start()) for m in BLOCK.finditer(text)]
    for i, (label, pos) in enumerate(blocks):
        end = blocks[i + 1][1] if i + 1 < len(blocks) else len(text)
        out[label] = VERIF.search(text, pos, end) is not None
    return out


def check_test(chapter, label, c):
    """Problems with a claim checked by Veriphy's sage_test fallback."""
    probs = []
    if c.get("level") not in ("kernel-test", "oracle-test"):
        probs.append(f"{chapter}: {label}: test {c['test']!r} needs `level: kernel-test` or `level: oracle-test`")
    if not isinstance(c.get("instances"), int) or c["instances"] < 3:
        probs.append(f"{chapter}: {label}: test {c['test']!r} needs `instances` of at least 3")
    if not os.path.exists(os.path.join(TESTS, c["test"] + ".json")):
        probs.append(f"{chapter}: {label}: test record .veriphy/tests/{c['test']}.json missing")
    return probs


def check(chapter, decls, known, strict=False):
    probs = []
    qmd = os.path.join(TUT, chapter + ".qmd")
    man = yaml.safe_load(open(os.path.join(MAN, chapter + ".yaml"))) or {}
    text = "\n".join(fold_blocks.unfold(open(qmd, encoding="utf-8").read().split("\n")))
    labels = [m.group(1) for m in BLOCK.finditer(text)]
    shown_by = shown(text)
    callout = has_callout(text)
    entries = man.get("blocks", [])
    seen = {}
    for e in entries:
        seen[e["label"]] = seen.get(e["label"], 0) + 1
    for l in labels:
        if l not in seen:
            probs.append(f"{chapter}: block {l} has no manifest entry")
    for l, n in seen.items():
        if l not in labels:
            probs.append(f"{chapter}: manifest entry {l} is not a block of the chapter")
        if n > 1:
            probs.append(f"{chapter}: block {l} has {n} manifest entries")
    for e in entries:
        l = e["label"]
        if ("claims" in e) == ("none" in e):
            probs.append(f"{chapter}: {l} needs exactly one of `claims` or `none`")
            continue
        if "claims" in e and not callout.get(l):
            probs.append(f"{chapter}: {l}: a block with claims needs a Verification callout")
        for c in e.get("claims", []):
            name, rec = c.get("lean"), c.get("recipe")
            if c.get("test"):
                probs += check_test(chapter, l, c)
            if strict and level(c, known) == "pending":
                probs.append(f"{chapter}: {l}: claim {c.get('claim')!r} is pending (no recipe certificate and no test)")
            if not name or name not in decls:
                probs.append(f"{chapter}: {l}: Lean declaration {name!r} not found")
            if rec == "mathlib":
                if not c.get("why"):
                    probs.append(f"{chapter}: {l}: recipe mathlib without `why`")
            elif rec not in known:
                probs.append(f"{chapter}: {l}: unknown recipe {rec!r}")
            elif not c.get("certificate") or not os.path.exists(os.path.join(CERTS, c["certificate"])):
                probs.append(f"{chapter}: {l}: certificate {c.get('certificate')!r} missing under .veriphy/certificates/")
            if name and name not in shown_by.get(l, set()):
                probs.append(f"{chapter}: {l}: {name} is not shown in the block's Verification callout")
    return probs, len(labels), sum(1 for e in entries if "claims" in e or "none" in e)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--coverage" in sys.argv:
        for q in sorted(glob.glob(os.path.join(TUT, "part-*.qmd"))):
            ch = os.path.basename(q)[:-4]
            n = len(BLOCK.findall(open(q, encoding="utf-8").read()))
            m = os.path.join(MAN, ch + ".yaml")
            k = len((yaml.safe_load(open(m)) or {}).get("blocks", [])) if os.path.exists(m) else 0
            print(f"{ch}: {k}/{n} blocks in the manifest")
        sys.exit(0)
    if "--levels" in sys.argv:
        known, total = recipes(), {k: 0 for k in LEVELS}
        for m in sorted(glob.glob(os.path.join(MAN, "part-*.yaml"))):
            counts = {k: 0 for k in LEVELS}
            for e in (yaml.safe_load(open(m)) or {}).get("blocks", []):
                for c in e.get("claims", []):
                    counts[level(c, known)] += 1
            for k in LEVELS:
                total[k] += counts[k]
            print(os.path.basename(m)[:-5] + ": " + ", ".join(f"{k} {counts[k]}" for k in LEVELS))
        print("total: " + ", ".join(f"{k} {total[k]}" for k in LEVELS))
        sys.exit(0)
    chapters = args or sorted(os.path.basename(m)[:-5] for m in glob.glob(os.path.join(MAN, "part-*.yaml")))
    decls, known, allp = declarations(), recipes(), []
    for ch in chapters:
        p, n, k = check(ch, decls, known, "--strict" in sys.argv)
        allp += p
        print(f"{ch}: {k}/{n} blocks covered; problems {len(p)}")
    for p in allp:
        print("  " + p)
    sys.exit(1 if allp else 0)
