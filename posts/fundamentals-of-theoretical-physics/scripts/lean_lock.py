"""Run a Lean command under the shared Lean lock: at most SLOTS Lean processes at once, and only with memory free.

The machine has 16 GB and a Lean process importing mathlib uses about 4.5 GB (much of it the shared, memory-mapped
mathlib .olean files). Every `lake build`, `lake env lean` and `.veriphy/certify.py` run of the certification goes
through this lock, so parallel agents never exhaust memory. A slot is a directory `.lake/lean-slots/<k>` holding the
owner's pid; a slot whose owner has died is reclaimed.

Usage: python3 scripts/lean_lock.py <command> [args...]     (exit code of the command)
"""
import os, subprocess, sys, time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SLOTS = int(os.environ.get("KK_LEAN_SLOTS", "3"))
DIR = os.path.join(ROOT, "lean", ".lake", "lean-slots")
MIN_FREE_GB = float(os.environ.get("KK_LEAN_MIN_FREE_GB", "5"))


def _free_gb():
    """Free plus reclaimable memory, in GB (macOS vm_stat; 0 when unknown)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    except OSError:
        return 0.0
    page = 16384
    pages = {}
    for line in out.splitlines()[1:]:
        k, _, v = line.partition(":")
        try:
            pages[k.strip()] = int(v.strip().rstrip("."))
        except ValueError:
            pass
    keys = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    return sum(pages.get(k, 0) for k in keys) * page / 2 ** 30


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def acquire():
    """Take a free slot (waiting for one and for free memory); return its directory."""
    os.makedirs(DIR, exist_ok=True)
    while True:
        for k in range(SLOTS):
            slot = os.path.join(DIR, str(k))
            pidf = os.path.join(slot, "pid")
            if os.path.isdir(slot):
                try:
                    pid = int(open(pidf).read())
                except (OSError, ValueError):
                    pid = -1
                if pid > 0 and _alive(pid):
                    continue
                try:
                    young = time.time() - os.path.getmtime(slot) < 10
                except OSError:  # removed meanwhile by another waiter
                    continue
                if pid == -1 and young:
                    continue  # being created
                try:
                    if os.path.exists(pidf):
                        os.remove(pidf)
                    os.rmdir(slot)
                except OSError:
                    continue
            try:
                os.mkdir(slot)
            except FileExistsError:
                continue
            if _free_gb() < MIN_FREE_GB and any(os.path.isdir(os.path.join(DIR, str(j))) for j in range(SLOTS) if j != k):
                os.rmdir(slot)  # not enough memory while another Lean runs: wait
                break
            with open(pidf, "w") as f:
                f.write(str(os.getpid()))
            return slot
        time.sleep(3)


def release(slot):
    try:
        os.remove(os.path.join(slot, "pid"))
        os.rmdir(slot)
    except OSError:
        pass


def run(cmd, **kw):
    """subprocess.run(cmd, **kw) under the lock."""
    slot = acquire()
    try:
        return subprocess.run(cmd, **kw)
    finally:
        release(slot)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    env = dict(os.environ, PATH=os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"])
    sys.exit(run(sys.argv[1:], env=env).returncode)
