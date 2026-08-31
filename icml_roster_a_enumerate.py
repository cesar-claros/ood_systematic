"""ICML protocol roster A: checkpoint enumeration (HPC; metadata only;
frozen protocol section 8.2 + the P-4 denominator rule).

Lists, WITHOUT loading or forwarding anything:
- all OpenOOD CIFAR-10 and CIFAR-100 ResNet-18 base cross-entropy runs
  under the declared roots (best*.ckpt discovery, stage-3 convention);
- every ImageNet-200 ResNet-18 checkpoint under the declared root,
  EXCLUDING the three inspected base CE pilot runs (by path substring);
- (D-R4 prep, informational only) candidate ViT experiment directories
  under EXPERIMENT_ROOT_DIR.

Writes the enumeration manifest with a sha256. PROTOCOL GATE: commit the
manifest BEFORE any forward pass; the extractors refuse to run without
it. No outcome of any kind is read here.

Usage (HPC, inside the container, from code/):
    python icml_roster_a_enumerate.py \
        --cifar10_root  $DATASET_ROOT_DIR/openood/results/cifar10_res18 \
        --cifar100_root $DATASET_ROOT_DIR/openood/results/cifar100_res18 \
        --in200_root    $DATASET_ROOT_DIR/openood/results
Output: pilot0/icml_roster_a_manifest.json  (rsync back and commit)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

INSPECTED_IN200 = (
    "imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0",
    "imagenet200_resnet18_224x224_base_e90_lr0.1_default/s1",
    "imagenet200_resnet18_224x224_base_e90_lr0.1_default/s2",
)


def dedup_by_parent(paths: list[str]) -> list[str]:
    """One checkpoint per run dir (the stage-3 discover_ckpts rule):
    OpenOOD saves best.ckpt AND its named best_epoch* twin per seed;
    both are the same run, and E1's checkpoint-cluster bootstrap must
    see each run once. Prefer best.ckpt, else the first sorted file."""
    by_parent: dict[str, str] = {}
    for p in sorted(paths):
        parent = str(Path(p).parent)
        if parent not in by_parent or Path(p).name == "best.ckpt":
            by_parent[parent] = p
    return [by_parent[k] for k in sorted(by_parent)]


def find_ckpts(root: Path) -> list[str]:
    out = []
    if root and root.is_dir():
        for p in sorted(root.rglob("best*.ckpt")):
            out.append(str(p))
    return dedup_by_parent(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar10_root", type=str, default=None)
    ap.add_argument("--cifar100_root", type=str, default=None)
    ap.add_argument("--in200_root", type=str, default=None)
    args = ap.parse_args()

    manifest: dict = {"protocol": "ICML retarget section 8.2 roster A",
                      "rule": "metadata-only enumeration committed before "
                              "any forward pass; IN200 excludes the three "
                              "inspected base CE runs", "rosters": {}}
    for name, root in (("cifar10", args.cifar10_root),
                       ("cifar100", args.cifar100_root)):
        cks = find_ckpts(Path(root)) if root else []
        manifest["rosters"][name] = cks
        print(f"[enum] {name}: {len(cks)} checkpoints")
    in200 = []
    if args.in200_root:
        for p in find_ckpts(Path(args.in200_root)):
            if "imagenet200" not in p:
                continue
            if any(x in p for x in INSPECTED_IN200):
                continue
            in200.append(p)
    manifest["rosters"]["imagenet200_extra"] = in200
    manifest["excluded_inspected_in200"] = list(INSPECTED_IN200)
    print(f"[enum] imagenet200_extra: {len(in200)} checkpoints "
          f"(3 inspected runs excluded)")

    vits = []
    exp_root = os.environ.get("EXPERIMENT_ROOT_DIR")
    if exp_root:
        for p in sorted(Path(exp_root).glob("**/")):
            nm = p.name.lower()
            if "vit" in nm and (p / "hydra").is_dir():
                vits.append(str(p.relative_to(exp_root)))
                if len(vits) >= 200:
                    break
    manifest["dr4_vit_candidates_informational"] = vits
    print(f"[enum] D-R4 ViT candidate dirs (informational): {len(vits)}")

    text = json.dumps(manifest, indent=1)
    out = Path("pilot0/icml_roster_a_manifest.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    print(f"[enum] wrote {out}; sha256 "
          f"{hashlib.sha256(text.encode()).hexdigest()}")
    print("[enum] PROTOCOL GATE: rsync back and COMMIT this manifest "
          "before running any roster-A extractor.")


if __name__ == "__main__":
    main()
