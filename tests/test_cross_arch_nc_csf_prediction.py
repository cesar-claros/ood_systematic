"""Tests for cross-architecture NC-to-CSF prediction helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cross_arch_nc_csf_prediction import (
    Example,
    ModelKey,
    evaluate_recommendations,
    normalize_method,
    read_ood_groups,
    should_skip_method,
)


class CrossArchNcCsfPredictionTests(unittest.TestCase):
    """Unit tests for parsing and metric helpers."""

    def test_method_filtering_matches_statistical_pipeline(self) -> None:
        """Projection variants are removed except PCA/KPCA global variants."""
        self.assertTrue(should_skip_method("GEN global", True, False))
        self.assertTrue(should_skip_method("Energy class pred", True, False))
        self.assertFalse(should_skip_method("KPCA RecError global", True, False))
        self.assertTrue(should_skip_method("CTMmean", True, False))
        self.assertTrue(should_skip_method("Confidence", True, True))
        self.assertEqual(normalize_method("KPCA RecError global"), "KPCA RecError")

    def test_read_ood_groups_from_two_row_clip_header(self) -> None:
        """CLIP grouping files have two header rows and numeric group labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_file = Path(tmpdir) / "clip_distances_cifar10.csv"
            clip_file.write_text(
                ",global,global,class-aware,class-aware,group\n"
                ",kid mean,fid,inv text alignment mean,img centroid dist mean,\n"
                "test,0,0,0,0,0\n"
                "cifar100,0,0,0,0,1\n"
                "isun,0,0,0,0,2\n"
                "textures,0,0,0,0,3\n",
                encoding="utf-8",
            )

            groups = read_ood_groups(clip_file)

        self.assertEqual(groups["near"], ["cifar100"])
        self.assertEqual(groups["mid"], ["isun"])
        self.assertEqual(groups["far"], ["textures"])
        self.assertEqual(groups["all"], ["cifar100", "isun", "textures"])

    def test_evaluate_recommendations_computes_hits_and_regret(self) -> None:
        """Evaluation uses the per-example oracle score as regret reference."""
        key_a = ModelKey("cifar10", "ResNet18", "confidnet", "False", 1, 2.2)
        key_b = ModelKey("cifar100", "ResNet18", "confidnet", "False", 1, 2.2)
        examples = [
            Example(key_a, (0.1,) * 8, "A", {"A": 1.0, "B": 2.0}),
            Example(key_b, (0.2,) * 8, "C", {"A": 1.5, "C": 1.0}),
        ]

        metrics, rows = evaluate_recommendations(
            examples,
            predicted_labels=["A", "A"],
            predicted_topk=[["A", "B"], ["A", "B"]],
        )

        self.assertAlmostEqual(metrics["top1_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["topk_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mean_regret"], 0.25)
        self.assertEqual(rows[0]["top1_hit"], 1)
        self.assertEqual(rows[1]["top1_hit"], 0)


if __name__ == "__main__":
    unittest.main()
