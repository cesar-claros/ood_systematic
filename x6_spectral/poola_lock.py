"""X6 prediction-lock manifest: generate and verify (pass-5.1 audit protocol).

The pass-5 re-review (5.1) established that a gitignored prediction directory
plus a user-supplied boolean is a self-attestation, not an auditable lock.
This module replaces it. The lock artifact is a TRACKED manifest recording,
at generation time: the SHA-256 and cell key of every prediction JSON, the
SHA-256 of every consumed feature file, the SHA-256 of the campaign scripts,
the git HEAD, and a UTC timestamp. Committing the manifest and tagging that
commit IS the lock; the prediction JSONs themselves may stay HPC-side
because their hashes are tracked.

poola_outcomes imports verify_lock and refuses to generate outcomes unless:
  - every manifest-listed prediction file exists with a matching hash;
  - no unlisted prediction file is present in the prediction directory;
  - every listed feature file matches its recorded hash (no swap between
    measurement and outcome generation);
  - the manifest file is committed, unmodified, and the expected tag exists
    and contains this exact manifest content (git checks skippable only for
    the synthetic self-test, which says so loudly).

Usage (from code/):
    python x6_spectral/poola_lock.py generate --pool l14 \
        --features-dir $EXPERIMENT_ROOT_DIR/pool_a/features
    python x6_spectral/poola_lock.py verify --pool l14 \
        --features-dir $EXPERIMENT_ROOT_DIR/pool_a/features --expect-tag x6-l14-lock
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

X6_DIR = Path(__file__).resolve().parent
SCRIPTS = ["poola_measure.py", "poola_outcomes.py", "poola_score.py",
           "poola_lock.py", "spectra_campaign_harness.py"]
POOL_DIRS = {"main": "outputs/poola", "l14": "outputs/poola_l14"}
DEFAULT_TAG = {"l14": "x6-l14-lock"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=X6_DIR)


def pred_dir_for(pool: str) -> Path:
    return X6_DIR / POOL_DIRS[pool]


def manifest_path_for(pool: str) -> Path:
    return pred_dir_for(pool) / "manifest.json"


def generate(pool: str, features_dir: Path, rule_version: str) -> Path:
    pred_dir = pred_dir_for(pool)
    preds = sorted(pred_dir.glob("*__*.json"))
    if not preds:
        sys.exit(f"no prediction JSONs in {pred_dir}; nothing to lock")
    entries = []
    for p in preds:
        rec = json.load(open(p))
        entries.append({"file": p.name, "sha256": sha256(p),
                        "cell": rec.get("cell")})
    feats = [{"file": f.name, "sha256": sha256(f)}
             for f in sorted(features_dir.glob("*.npz"))]
    scripts = [{"file": s, "sha256": sha256(X6_DIR / s)} for s in SCRIPTS]
    manifest = {
        "pool": pool, "rule_version": rule_version,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head_at_generation": git("rev-parse", "HEAD").stdout.strip(),
        "n_predictions": len(entries), "predictions": entries,
        "features": feats, "scripts": scripts,
    }
    out = manifest_path_for(pool)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"manifest: {len(entries)} predictions, {len(feats)} feature "
          f"files, {len(scripts)} scripts -> {out}")
    print("LOCK PROTOCOL: commit this manifest, then tag the commit "
          f"(suggested tag: {DEFAULT_TAG.get(pool, 'x6-lock')}); only then "
          "run poola_outcomes.py.")
    return out


def verify_lock(pool: str, features_dir: Path | None,
                expect_tag: str | None, git_checks: bool = True,
                pred_dir: Path | None = None,
                manifest_path: Path | None = None) -> list[str]:
    """Return a list of problems; empty means the lock verifies."""
    pred_dir = pred_dir or pred_dir_for(pool)
    manifest_path = manifest_path or (pred_dir / "manifest.json")
    problems: list[str] = []
    if not manifest_path.exists():
        return [f"manifest missing: {manifest_path}"]
    manifest = json.load(open(manifest_path))
    listed = {e["file"] for e in manifest["predictions"]}
    for e in manifest["predictions"]:
        p = pred_dir / e["file"]
        if not p.exists():
            problems.append(f"prediction missing: {e['file']}")
        elif sha256(p) != e["sha256"]:
            problems.append(f"prediction hash mismatch: {e['file']}")
    extra = {p.name for p in pred_dir.glob("*__*.json")} - listed
    if extra:
        problems.append(f"{len(extra)} prediction files not in manifest "
                        f"(e.g. {sorted(extra)[:3]})")
    if features_dir is not None:
        for e in manifest.get("features", []):
            f = features_dir / e["file"]
            if not f.exists():
                problems.append(f"feature file missing: {e['file']}")
            elif sha256(f) != e["sha256"]:
                problems.append(f"feature hash mismatch: {e['file']} "
                                "(features changed since measurement)")
    if git_checks:
        rel = git("ls-files", "--full-name", "--error-unmatch",
                  str(manifest_path))
        if rel.returncode != 0:
            problems.append("manifest is not tracked by git")
        else:
            relpath = rel.stdout.strip()
            if git("status", "--porcelain", "--",
                   str(manifest_path)).stdout.strip():
                problems.append("manifest has uncommitted changes")
            if not expect_tag:
                problems.append("no expected tag supplied (--expect-tag)")
            else:
                tag = git("rev-list", "-n", "1", expect_tag)
                if tag.returncode != 0:
                    problems.append(f"tag not found: {expect_tag}")
                else:
                    shown = git("show", f"{expect_tag}:{relpath}")
                    if shown.returncode != 0:
                        problems.append(f"tag {expect_tag} does not contain "
                                        f"{relpath}")
                    elif shown.stdout != open(manifest_path).read():
                        problems.append(f"manifest at tag {expect_tag} "
                                        "differs from working copy")
    else:
        print("NOTE: git checks skipped (hash-only verification; valid only "
              "for the synthetic self-test)")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="X6 prediction-lock manifest")
    ap.add_argument("mode", choices=["generate", "verify"])
    ap.add_argument("--pool", choices=list(POOL_DIRS), default="l14")
    ap.add_argument("--features-dir", type=str, default="pool_a_features")
    ap.add_argument("--rule-version", type=str, default="r8+pass5.1")
    ap.add_argument("--expect-tag", type=str, default=None)
    args = ap.parse_args()
    features_dir = Path(args.features_dir)
    if args.mode == "generate":
        generate(args.pool, features_dir, args.rule_version)
        return
    tag = args.expect_tag or DEFAULT_TAG.get(args.pool)
    problems = verify_lock(args.pool, features_dir, tag)
    if problems:
        print("LOCK VERIFICATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"lock verifies: manifest + {tag} + feature hashes all consistent")


if __name__ == "__main__":
    main()
