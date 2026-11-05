"""Report mismatches between the installed tools and the pinned versions (VERSIONS, lean-toolchain).

Checks `lean --version` against lean-toolchain, `sage --version` and `sage -pip show veriphy` against VERSIONS,
`quarto --version` against VERSIONS (major.minor), and that `rsvg-convert` (SVG figures in the PDF) is available.
Exit code 1 on any mismatch or missing tool.
"""
import os, re, shutil, subprocess, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def pinned():
    out = {}
    for line in open(os.path.join(ROOT, "VERSIONS")):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main():
    want, bad = pinned(), []
    tc = open(os.path.join(ROOT, "lean-toolchain")).read().strip().split(":")[-1]
    lean = run(["lean", "--version"])
    if f"version {tc.lstrip('v')}" not in lean:
        bad.append(f"lean: pinned {tc}, installed {lean or 'not found'}")
    sage = run([os.environ.get("SAGE", "sage"), "--version"])
    if f"version {want['sage']}" not in sage:
        bad.append(f"sage: pinned {want['sage']}, installed {sage or 'not found'}")
    ver = re.search(r"Version: (\S+)", run([os.environ.get("SAGE", "sage"), "-pip", "show", "veriphy"]))
    if not ver or ver.group(1) != want["veriphy"]:
        bad.append(f"veriphy daemon: pinned {want['veriphy']}, installed {ver.group(1) if ver else 'not found'}")
    quarto = run(["quarto", "--version"])
    if not quarto.startswith(want["quarto"]):
        bad.append(f"quarto: pinned {want['quarto']}.x, installed {quarto or 'not found'}")
    if not shutil.which("rsvg-convert"):
        bad.append("rsvg-convert: not found (needed to place SVG figures in the PDF)")
    for b in bad:
        print(f"[version mismatch] {b}")
    print(f"versions: lean {tc}, sage {want['sage']}, quarto {want['quarto']}, veriphy {want['veriphy']}; problems {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
