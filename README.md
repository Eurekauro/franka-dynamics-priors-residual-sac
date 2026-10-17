# Franka Dynamics Priors & Residual SAC

This is the final, portable project layout for the **Sim-trained** 7-DOF
Franka Panda inverse-dynamics-prior study.  It preserves three frozen priors
(MLP, Expanded GP, Expanded RBF), six Residual SAC control arms, FER-QP V2.1
safety, P4-B saved policies, and P5 evidence.

The strict P5 result covers the original five arms under a nominal-only safety
controller.  The PD+Gravity extension is retained separately as a post-hoc
exploratory baseline.  This repository is simulation evidence only; it is not
real-robot, FCI, ISO, or hardware safety certification.

## Quick verification

Create a Python 3.11 environment from `requirements.lock`, then run:

```bash
PYTHONPATH=src python tools/verify_migration.py
```

The command validates model/policy hashes, safe NPZ loading, the MuJoCo asset,
Gymnasium reset/step, a NaN fail-closed path, and saved-result readability. It
does not train or re-run P4/P5.

## Boundaries

`internal_archive/provenance/` records source-to-destination hashes, exclusions
and licenses.  Original upstream/official data, AGPL LIP code, historical
training/replay artifacts and unresolved redistribution assets are deliberately
not public-release content.  See `docs/LIMITATIONS.md` and
`internal_archive/provenance/PROVENANCE_CLASSIFICATION_REPORT.md`.
