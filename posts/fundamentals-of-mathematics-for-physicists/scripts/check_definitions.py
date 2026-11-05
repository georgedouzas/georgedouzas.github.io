"""Check the definition blocks of the tutorial: one notion per block (no stacked titles), enough example blocks
after each definition before the next definition or section (ten in a raised chapter or for a new definition, three
otherwise; scripts/density.py), and at most six definitions per `## ` section (design D19; specs/tutorial-content).

A definition is `::: {#def-...}` with a `#### Title` line; a title naming several notions (containing ',' or ' and ')
is reported as stacked. Examples are `::: {#exm-...}` blocks. Figures (`![...](figures/...)`) are counted and
reported per chapter.

Usage: python3 check_definitions.py [--table]        exit code 1 on any problem; --table prints the review table
"""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fold_blocks  # the reading wrappers of change kk-csdr-presentation are ignored
import density  # change kk-csdr-ten-examples: ten examples per statement in raised chapters

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TUTORIAL = ROOT
BLOCK = re.compile(r"^:::+ \{#((?:def|exm|prp|thm|lem|cor|exr)-[A-Za-z0-9-]+)[^}]*\}\n(?:#### (.*)\n)?", re.M)
FIG = re.compile(r"!\[.*?\]\(figures/[\w.-]+\)", re.S)


def review():
    rows, problems = [], []
    for f in sorted(glob.glob(os.path.join(TUTORIAL, "part-*.qmd"))):
        text = "\n".join(fold_blocks.unfold(open(f, encoding="utf-8").read().split("\n")))
        name = os.path.basename(f)
        blocks = [(m.group(1), (m.group(2) or "").strip(), m.start()) for m in BLOCK.finditer(text)]
        nfig = len(FIG.findall(text))
        defs = [b for b in blocks if b[0].startswith("def-")]
        for k, (label, title, pos) in enumerate(defs):
            # examples between this definition and the next definition
            nxt = defs[k + 1][2] if k + 1 < len(defs) else len(text)
            n_ex = sum(1 for b in blocks if b[0].startswith("exm-") and pos < b[2] < nxt)
            stacked = ("," in title) or (" and " in title.lower()) or (" or " in title.lower())
            rows.append((name, label, title, n_ex, stacked))
            if stacked:
                problems.append(("one notion per definition", f"{name}: {label} ('{title}') names several notions"))
            need = density.minimum(name, label)
            if n_ex < need:
                problems.append(("examples per definition", f"{name}: {label} has {n_ex} example(s), needs {need}"))
        if "To be written" not in text and nfig == 0:
            problems.append(("figure per chapter", f"{name}: no figure"))
        # design D19: the limit of six definitions applies per `## ` section of a group chapter
        secs = [(m.group(1).strip(), m.start()) for m in re.finditer(r"^## (.*)$", text, re.M)]
        bounds = [(t, st, (secs[i + 1][1] if i + 1 < len(secs) else len(text))) for i, (t, st) in enumerate(secs)] or [("(whole file)", 0, len(text))]
        for t, st, en in bounds:
            # a nested part of a main-theme chapter (design D27) is recognised by its own `### Exercises`; its unit is
            # then the `### ` section
            units = [(t, st, en)]
            if re.search(r"^### Exercises", text[st:en], re.M):
                subs = [(m.group(1).strip(), st + m.start()) for m in re.finditer(r"^### (.*)$", text[st:en], re.M)]
                units = [(u, a, (subs[i + 1][1] if i + 1 < len(subs) else en)) for i, (u, a) in enumerate(subs)]
            for ut, us, ue in units:
                n_defs = sum(1 for b in defs if us <= b[2] < ue)
                if n_defs > 6:
                    problems.append(("at most six definitions per section", f"{name} / {ut}: {n_defs} definitions"))
    return rows, problems


if __name__ == "__main__":
    rows, problems = review()
    if "--table" in sys.argv:
        print("| chapter | definition | title | examples | stacked |\n|---|---|---|---|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {'yes' if r[4] else ''} |")
    for n, msg in problems:
        print(f"[{n}] {msg}")
    print(f"{len(rows)} definitions; problems {len(problems)}")
    sys.exit(1 if problems else 0)
