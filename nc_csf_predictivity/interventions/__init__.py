"""Pilot 1 intervention analysis (code/pilot1/MANIFEST.md).

Blinded-stage tooling: `extract_manipulation.py` (HPC) computes geometry
and nuisance vectors from train/val splits only -- it never forwards OOD
data -- and `manipulation_report.py` (local) renders the M1/M2 gate report
that must be committed before any CSF scoring.
"""
