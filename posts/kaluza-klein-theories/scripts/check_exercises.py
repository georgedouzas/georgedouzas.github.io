"""Draft chapters under tutorial/_drafts/ (material not yet split into the numbered chapters) count as chapters here.
Check the exercises of the tutorial: every exercise has a folded solution, and every Sage solution has a
committed script with a recorded output.

Conventions (design D13): an exercise is a `::: {#exr-...}` block; its solution is the next block, a
`::: {.callout-note collapse="true" ...}` callout; a Sage solution is `{{< sage <label> >}}` with
sage/exercises/<label>.sage and <label>.out (written by `make exercises`); a Lean solution is `{{< lean <Name> >}}`
(checked by extract_statements.py).

Usage: python3 check_exercises.py        exit code 1 on any problem
"""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fold_blocks  # the reading wrappers of change kk-csdr-presentation are ignored

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TUTORIAL = ROOT
EX = os.path.join(ROOT, "sage-exercises")
EXR = re.compile(r"^:::+ \{#(exr-[A-Za-z0-9-]+)[^}]*\}\n.*?^:::+[ \t]*\n(.*?)(?=^:::+ \{#|\Z)", re.M | re.S)
SOLUTION = re.compile(r"^\s*:::+ \{\.callout-note[^}]*collapse=\"true\"[^}]*\}", re.M)
SAGE = re.compile(r"\{\{<\s*sage\s+([\w-]+)\s*>\}\}")


def problems():
    out, used = [], set()
    for f in sorted((glob.glob(os.path.join(TUTORIAL, "*.qmd")) + glob.glob(os.path.join(TUTORIAL, "_drafts", "*.qmd")))):
        text = "\n".join(fold_blocks.unfold(open(f, encoding="utf-8").read().split("\n")))
        for m in EXR.finditer(text):
            label, after = m.group(1), m.group(2)
            if not SOLUTION.match(after.lstrip("\n")):
                out.append(("exercise has a folded solution", f"{os.path.basename(f)}: {label} is not followed by a collapsed solution callout"))
        for label in SAGE.findall(text):
            used.add(label)
            if not os.path.exists(os.path.join(EX, label + ".sage")):
                out.append(("sage solution exists", f"{os.path.basename(f)}: no sage/exercises/{label}.sage"))
            elif not os.path.exists(os.path.join(EX, label + ".out")):
                out.append(("sage solution exists", f"{os.path.basename(f)}: no recorded output sage/exercises/{label}.out (run `make exercises`)"))
            elif os.path.getmtime(os.path.join(EX, label + ".out")) < os.path.getmtime(os.path.join(EX, label + ".sage")):
                out.append(("sage solution exists", f"sage/exercises/{label}.out is older than its script (run `make exercises`)"))
    for s in glob.glob(os.path.join(EX, "*.sage")):
        name = os.path.splitext(os.path.basename(s))[0]
        if name not in used:
            out.append(("sage solution is used", f"sage/exercises/{name}.sage is not used by any exercise"))
    return out


if __name__ == "__main__":
    found = problems()
    for n, msg in found:
        print(f"[{n}] {msg}")
    n_ex = sum(len(EXR.findall(open(f, encoding="utf-8").read())) for f in (glob.glob(os.path.join(TUTORIAL, "*.qmd")) + glob.glob(os.path.join(TUTORIAL, "_drafts", "*.qmd"))))
    print(f"{n_ex} exercises, {len(glob.glob(os.path.join(EX, '*.sage')))} Sage solutions; problems {len(found)}")
    sys.exit(1 if found else 0)
