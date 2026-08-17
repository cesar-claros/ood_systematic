"""Pilot 0: operator-level falsification harness for the X1+X3 flagship.

Consumes per-checkpoint feature caches produced by `extract_pilot0.py`
(HPC side) and runs the pre-registered operator study locally: head
rotations on frozen features, exact-mean X1 predictions, feature-side
invariance gates, H-estimator validation, and the AUGRC/failure-AUROC
identity check. See documentation/X1_X3_flagship_mechanistic_paper_plan.md
section 8 (Pilot 0) for the gates this package implements.
"""
