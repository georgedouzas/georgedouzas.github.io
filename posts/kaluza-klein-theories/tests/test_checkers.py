"""Tests of the verification levels in scripts/check_format.py and scripts/check_verification.py (design D29)."""
import os, sys, tempfile, unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import check_format, check_verification

SHAPE = "Verification = Veriphy, Lean, Sage"


def callout(veriphy):
    return ('::: {.callout-tip .verification collapse="true" title="Verification"}\n'
            f"**Veriphy.** {veriphy}\n\n**Lean.**\n\n{{{{< lean KaluzaKlein.x >}}}}\n\n**Sage.** The oracle script.\n:::\n")


def shape_problems(veriphy, strict):
    with tempfile.NamedTemporaryFile("w", suffix=".qmd", delete=False) as f:
        f.write("# Chapter\n\n::: {#prp-x}\n#### X\n\nA claim.\n:::\n\n" + callout(veriphy))
    argv = sys.argv
    sys.argv = ["check_format.py"] + (["--strict"] if strict else [])
    try:
        return [m for n, m in check_format.check_file(f.name) if n == SHAPE]
    finally:
        sys.argv = argv
        os.unlink(f.name)


class FormatLevels(unittest.TestCase):
    def test_certificate_passes_strict(self):
        self.assertEqual(shape_problems("Recipe `dimension`, certificate `la-dim.sage`.", True), [])

    def test_kernel_test_passes_strict(self):
        self.assertEqual(shape_problems("Test (kernel-checked, 5 instances); script `rank_nullity_test.sage`.", True), [])

    def test_oracle_test_passes_strict(self):
        self.assertEqual(shape_problems("Test (oracle, 20 instances); record `heat_trace.json`.", True), [])

    def test_pending_fails_strict_only(self):
        self.assertEqual(shape_problems("Pending.", False), [])
        self.assertEqual(len(shape_problems("Pending.", True)), 1)

    def test_no_level_fails_strict(self):
        self.assertEqual(len(shape_problems("Proved by hand.", True)), 1)


class ManifestLevels(unittest.TestCase):
    known = {"dimension"}

    def test_level_certificate(self):
        c = {"recipe": "dimension", "certificate": "la-dim.sage"}
        self.assertEqual(check_verification.level(c, self.known), "certificate")

    def test_level_tests(self):
        for lvl in ("kernel-test", "oracle-test"):
            c = {"recipe": "mathlib", "why": "general", "test": "t", "level": lvl, "instances": 3}
            self.assertEqual(check_verification.level(c, self.known), lvl)

    def test_level_pending(self):
        self.assertEqual(check_verification.level({"recipe": "mathlib", "why": "general"}, self.known), "pending")

    def test_test_needs_level_instances_and_record(self):
        probs = check_verification.check_test("ch", "thm-x", {"test": "no_such_record", "instances": 2})
        self.assertEqual(len(probs), 3)

    def test_definition_without_claim_has_no_callout(self):
        text = "::: {#def-x}\n#### X\n\nA notion.\n:::\n\n" + callout("Pending.")
        self.assertTrue(check_verification.has_callout(text)["def-x"])
        text = "::: {#def-x}\n#### X\n\nA notion.\n:::\n"
        self.assertFalse(check_verification.has_callout(text)["def-x"])


if __name__ == "__main__":
    unittest.main()
