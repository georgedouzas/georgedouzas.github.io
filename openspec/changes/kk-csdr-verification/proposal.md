# Proposal

## Why

The book *Kaluza-Klein Theories and Coset Space Dimensional Reduction* is to verify every claim it makes with Lean,
mathlib and Veriphy (Sage as an untrusted oracle). The content was planned and written under the change
`kk-csdr-tutorial`, which also began this verification: a pilot on Linear algebra and six certified chapters (Sets,
Linear algebra, Real analysis, Complex analysis, Differential equations, Fourier analysis; 4,111 claims). The author
paused the verification in October 2026 to finish the content first. This change carries the remaining verification
work so that the content change can be archived and the verification resumed later as one unit.

## What Changes

- Certify every claim of the remaining chapters, including the chapters added by `kk-csdr-tutorial` group 12
  (functional analysis, spinors and the Dirac operator, the Dirac equation, renormalisation, special relativity, the
  fermion parts of the reductions), whose blocks carry placeholder Verification boxes.
- Rebuild the Lean modules changed by the metric-signature switch (`AbelianKK/Jets` and `Weyl` exceed memory and must
  be split; `CSDR/Potential` has three proof errors) and import the CSDR modules.
- Complete the manifests `verification/<chapter>.yaml`, the gate (`check_format.py --strict`,
  `check_verification.py --strict`, the axiom audit, a Sage-free build) and the final integration checks.
- Feed the Veriphy gaps found (`verification/veriphy-gaps.md`) into Veriphy's next releases.

## Capabilities

### Modified Capabilities

- `result-verification`: the requirements on machine-checked claims, the manifests, Sage-free proofs, certificates
  and Veriphy coverage, moved here from `kk-csdr-tutorial`.

## Impact

Lean library `KaluzaKlein/`, `.veriphy/certificates/` and `.veriphy/tests/`, `verification/`, the Verification
boxes of the chapters. No change to the mathematics of the text except corrections the formalisation reveals.
