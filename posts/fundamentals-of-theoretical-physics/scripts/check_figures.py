"""Draft chapters under tutorial/_drafts/ (material not yet split into the numbered chapters) count as chapters here.
Check that every figure of the tutorial is produced by a committed Sage script, and every script's figure is used.

Convention: `figures/<name>.sage` writes `tutorial/figures/<name>.<svg|png>` (the script names that path) and prints
`[PASS] ...` lines for the quantities it plots. A chapter embeds it as `![caption](figures/<name>.svg){#fig-...}`;
the caption names the script.

Usage: python3 check_figures.py        exit code 1 on any problem
"""
import glob, os, re, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TUTORIAL = ROOT
SCRIPTS = os.path.join(ROOT, "figures")
IMG = re.compile(r"!\[(.*?)\]\((figures/[\w.-]+\.(?:svg|png|pdf))\)(\{[^}]*\})?", re.S)


def problems():
    out = []
    scripts = {os.path.splitext(os.path.basename(s))[0]: s for s in glob.glob(os.path.join(SCRIPTS, "*.sage"))}
    used = set()
    for f in sorted((glob.glob(os.path.join(TUTORIAL, "*.qmd")) + glob.glob(os.path.join(TUTORIAL, "_drafts", "*.qmd")))):
        for m in IMG.finditer(open(f, encoding="utf-8").read()):
            caption, path = m.group(1), m.group(2)
            name = os.path.splitext(os.path.basename(path))[0]
            used.add(name)
            if name not in scripts:
                out.append(("image has a script", f"{os.path.basename(f)}: {path} has no figures/{name}.sage"))
            elif os.path.basename(path) not in open(scripts[name]).read():
                out.append(("image has a script", f"figures/{name}.sage does not name its output {os.path.basename(path)}"))
            if not os.path.exists(os.path.join(TUTORIAL, path)):
                out.append(("image exists", f"{os.path.basename(f)}: {path} is missing (run `make figures`)"))
            if f"{name}.sage" not in caption:
                out.append(("caption names script", f"{os.path.basename(f)}: caption of {path} does not name {name}.sage"))
    for name in sorted(set(scripts) - used):
        out.append(("script figure is used", f"figures/{name}.sage produces a figure no chapter embeds"))
    return out


if __name__ == "__main__":
    found = problems()
    for n, msg in found:
        print(f"[{n}] {msg}")
    print(f"{len(glob.glob(os.path.join(SCRIPTS, '*.sage')))} figure scripts; problems {len(found)}")
    sys.exit(1 if found else 0)
