"""Cross-architecture prediction of competitive CSFs from NC metrics.

This script tests whether the 8 Papyan Neural Collapse metrics measured on a
trained model predict which confidence score function (CSF) obtains the lowest
OOD AUGRC. The intended evaluation trains on VGG13 configurations and tests on
ResNet18 configurations.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score


LOGGER = logging.getLogger(__name__)

PAPYAN_NC_METRICS = (
    "var_collapse",
    "equiangular_uc",
    "equiangular_wc",
    "equinorm_uc",
    "equinorm_wc",
    "max_equiangular_uc",
    "max_equiangular_wc",
    "self_duality",
)

DEFAULT_SOURCES = ("cifar10", "cifar100", "supercifar100", "tinyimagenet")
DEFAULT_STUDIES = ("confidnet", "devries", "dg")
GROUP_NAMES = {"1": "near", "2": "mid", "3": "far"}
META_COLUMNS = {"model", "drop out", "methods", "reward", "run", "test"}

METHOD_RENAMES = {
    "KPCA RecError global": "KPCA RecError",
    "PCA RecError global": "PCA RecError",
    "MCD-KPCA RecError global": "MCD-KPCA RecError",
    "MCD-PCA RecError global": "MCD-PCA RecError",
}
FILTER_EXCEPTIONS = set(METHOD_RENAMES)
EXCLUDED_MEAN_METHODS = {"CTMmean", "CTMmeanOC", "MCD-CTMmean", "MCD-CTMmeanOC"}


@dataclass(frozen=True)
class ModelKey:
    """Identifier shared by NC rows and CSF score rows."""

    dataset: str
    architecture: str
    study: str
    dropout: str
    run: int
    reward: float


@dataclass(frozen=True)
class Example:
    """Single model configuration with NC features and CSF AUGRC targets."""

    key: ModelKey
    features: tuple[float, ...]
    label: str
    method_scores: Mapping[str, float]

    @property
    def oracle_score(self) -> float:
        """Return the best available AUGRC for this model configuration."""
        return self.method_scores[self.label]


def normalize_dataset_name(dataset: str) -> str:
    """Map dataset aliases to the score-file naming convention."""
    return "supercifar100" if dataset == "supercifar" else dataset


def parse_dropout(value: str) -> str:
    """Convert score-file dropout values to NC-table boolean strings."""
    if value == "do1":
        return "True"
    if value == "do0":
        return "False"
    if value in {"True", "False"}:
        return value
    raise ValueError(f"Unknown dropout value: {value}")


def parse_reward(value: str) -> float:
    """Convert reward strings such as ``rew2.2`` to floats."""
    return float(value.replace("rew", ""))


def normalize_method(method: str) -> str:
    """Apply paper-facing method name aliases."""
    return METHOD_RENAMES.get(method.strip(), method.strip())


def should_skip_method(
    method: str,
    filter_method_variants: bool,
    drop_confidence: bool,
) -> bool:
    """Return whether a CSF should be excluded from the analysis."""
    raw_method = method.strip()
    renamed_method = normalize_method(raw_method)
    if drop_confidence and renamed_method == "Confidence":
        return True
    if renamed_method in EXCLUDED_MEAN_METHODS:
        return True
    if not filter_method_variants:
        return False
    has_projection_variant = (
        "global" in raw_method.lower() or "class" in raw_method.lower()
    )
    return has_projection_variant and raw_method not in FILTER_EXCEPTIONS


def read_nc_features(
    nc_file: Path,
    architecture: str,
    features: Sequence[str],
) -> dict[ModelKey, tuple[float, ...]]:
    """Read NC features keyed by model configuration."""
    keyed_features: dict[ModelKey, tuple[float, ...]] = {}
    with nc_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["architecture"] != architecture:
                continue
            key = ModelKey(
                dataset=normalize_dataset_name(row["dataset"]),
                architecture=row["architecture"],
                study=row["study"],
                dropout=parse_dropout(row["dropout"]),
                run=int(row["run"]),
                reward=parse_reward(row["reward"]),
            )
            keyed_features[key] = tuple(float(row[feature]) for feature in features)
    LOGGER.info(
        "Loaded %d NC rows for architecture=%s from %s",
        len(keyed_features),
        architecture,
        nc_file,
    )
    return keyed_features


def read_ood_groups(clip_file: Path) -> dict[str, list[str]]:
    """Read CLIP near/mid/far OOD groups from a two-row-header CSV."""
    groups: dict[str, list[str]] = {"near": [], "mid": [], "far": []}
    with clip_file.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        next(reader, None)
        for row in reader:
            if len(row) < 6 or row[0] == "test":
                continue
            group_name = GROUP_NAMES.get(row[5])
            if group_name is not None:
                groups[group_name].append(row[0])
    groups["all"] = [
        dataset for name in ("near", "mid", "far") for dataset in groups[name]
    ]
    return groups


def resolve_score_file(
    scores_dir: Path,
    metric: str,
    mcd: str,
    backbone_label: str,
    source: str,
) -> Path:
    """Find a score CSV, preferring per-run ``scores_all`` files."""
    labels: Iterable[str]
    if backbone_label == "auto":
        labels = sorted(
            {
                path.name.split(f"_MCD-{mcd}_", maxsplit=1)[1].rsplit(
                    f"_{source}.csv", maxsplit=1
                )[0]
                for path in scores_dir.glob(
                    f"scores*_{metric}_MCD-{mcd}_*_{source}.csv"
                )
            }
        )
    else:
        labels = (backbone_label,)

    for label in labels:
        for prefix in ("scores_all", "scores"):
            candidate = scores_dir / f"{prefix}_{metric}_MCD-{mcd}_{label}_{source}.csv"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No {metric} score file found for source={source}, mcd={mcd}, "
        f"backbone_label={backbone_label} in {scores_dir}"
    )


def numeric_mean(values: Iterable[str]) -> float | None:
    """Return the finite mean of CSV numeric strings, or ``None`` if empty."""
    parsed = []
    for value in values:
        if value == "":
            continue
        score = float(value)
        if math.isfinite(score):
            parsed.append(score)
    if not parsed:
        return None
    return float(np.mean(parsed))


def selected_score_columns(
    fieldnames: Sequence[str],
    source: str,
    group: str,
    clip_dir: Path,
) -> list[str]:
    """Return score columns for a source dataset and OOD group."""
    available = [name for name in fieldnames if name not in META_COLUMNS]
    if group == "all":
        return available

    clip_file = clip_dir / f"clip_distances_{source}.csv"
    if not clip_file.exists():
        raise FileNotFoundError(f"Missing CLIP group file: {clip_file}")
    groups = read_ood_groups(clip_file)
    return [name for name in groups[group] if name in available]


def read_competitive_labels(
    scores_dir: Path,
    architecture: str,
    backbone_label: str,
    metric: str,
    mcd: str,
    sources: Sequence[str],
    group: str,
    clip_dir: Path,
    filter_method_variants: bool,
    drop_confidence: bool,
) -> dict[ModelKey, dict[str, float]]:
    """Read mean group AUGRC per CSF for every model configuration."""
    scores_by_key: dict[ModelKey, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for source in sources:
        score_file = resolve_score_file(
            scores_dir, metric, mcd, backbone_label, source
        )
        with score_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Empty score file: {score_file}")
            score_columns = selected_score_columns(
                reader.fieldnames, source, group, clip_dir
            )
            if not score_columns:
                raise ValueError(f"No score columns for group={group} in {score_file}")

            for row in reader:
                raw_method = row["methods"]
                if should_skip_method(
                    raw_method, filter_method_variants, drop_confidence
                ):
                    continue
                score = numeric_mean(row[column] for column in score_columns)
                if score is None:
                    continue
                key = ModelKey(
                    dataset=source,
                    architecture=architecture,
                    study=row["model"],
                    dropout=parse_dropout(row["drop out"]),
                    run=int(row["run"]),
                    reward=parse_reward(row["reward"]),
                )
                scores_by_key[key][normalize_method(raw_method)].append(score)

        LOGGER.info("Loaded score rows from %s", score_file)

    collapsed: dict[ModelKey, dict[str, float]] = {}
    for key, method_values in scores_by_key.items():
        collapsed[key] = {
            method: float(np.mean(values)) for method, values in method_values.items()
        }
    LOGGER.info(
        "Built labels for %d model configurations from %s",
        len(collapsed),
        scores_dir,
    )
    return collapsed


def build_examples(
    nc_features: Mapping[ModelKey, tuple[float, ...]],
    method_scores: Mapping[ModelKey, Mapping[str, float]],
    studies: set[str],
) -> list[Example]:
    """Join NC features and CSF scores into supervised examples."""
    examples = []
    missing_nc = 0
    for key, scores in method_scores.items():
        if studies and key.study not in studies:
            continue
        features = nc_features.get(key)
        if features is None:
            missing_nc += 1
            continue
        label = min(scores, key=scores.get)
        examples.append(
            Example(
                key=key,
                features=features,
                label=label,
                method_scores=dict(scores),
            )
        )
    LOGGER.info(
        "Joined %d examples; skipped %d score configurations without NC rows",
        len(examples),
        missing_nc,
    )
    return examples


def make_classifier(random_state: int) -> RandomForestClassifier:
    """Create the NC-to-CSF classifier used in all evaluations."""
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def top_k_from_probabilities(
    classes: Sequence[str],
    probabilities: np.ndarray,
    top_k: int,
) -> list[list[str]]:
    """Return top-k class labels for every probability row."""
    classes_array = np.asarray(classes)
    top_k = min(top_k, len(classes_array))
    top_indices = np.argsort(-probabilities, axis=1)[:, :top_k]
    return [[str(label) for label in classes_array[row]] for row in top_indices]


def evaluate_recommendations(
    examples: Sequence[Example],
    predicted_labels: Sequence[str],
    predicted_topk: Sequence[Sequence[str]],
) -> tuple[dict[str, float], list[dict[str, str | float | int]]]:
    """Evaluate predicted CSF recommendations against oracle AUGRC labels."""
    y_true = [example.label for example in examples]
    y_pred = list(predicted_labels)
    labels = sorted(set(y_true) | set(y_pred))
    rows: list[dict[str, str | float | int]] = []
    regrets = []
    norm_regrets = []
    missing_predicted_scores = 0
    topk_hits = []

    for example, predicted, topk in zip(examples, y_pred, predicted_topk):
        selected_score = example.method_scores.get(predicted)
        if selected_score is None:
            missing_predicted_scores += 1
            regret = math.nan
            norm_regret = math.nan
        else:
            regret = selected_score - example.oracle_score
            norm_regret = regret / example.oracle_score if example.oracle_score else 0.0
            regrets.append(regret)
            norm_regrets.append(norm_regret)
        topk_hit = int(example.label in set(topk))
        topk_hits.append(topk_hit)
        rows.append(
            {
                "dataset": example.key.dataset,
                "architecture": example.key.architecture,
                "study": example.key.study,
                "dropout": example.key.dropout,
                "run": example.key.run,
                "reward": example.key.reward,
                "true_method": example.label,
                "predicted_method": predicted,
                "predicted_topk": "|".join(topk),
                "oracle_score": example.oracle_score,
                "predicted_score": (
                    selected_score if selected_score is not None else math.nan
                ),
                "regret": regret,
                "norm_regret": norm_regret,
                "top1_hit": int(predicted == example.label),
                "topk_hit": topk_hit,
            }
        )

    metrics = {
        "top1_accuracy": float(
            np.mean([true == pred for true, pred in zip(y_true, y_pred)])
        ),
        "topk_accuracy": float(np.mean(topk_hits)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "mean_regret": float(np.mean(regrets)) if regrets else math.nan,
        "median_regret": float(np.median(regrets)) if regrets else math.nan,
        "mean_norm_regret": float(np.mean(norm_regrets)) if norm_regrets else math.nan,
        "missing_predicted_score_count": float(missing_predicted_scores),
    }
    return metrics, rows


def train_and_predict(
    train_examples: Sequence[Example],
    test_examples: Sequence[Example],
    top_k: int,
    random_state: int,
) -> tuple[
    dict[str, float],
    list[dict[str, str | float | int]],
    RandomForestClassifier,
]:
    """Fit a classifier on train examples and evaluate on test examples."""
    x_train = np.asarray([example.features for example in train_examples], dtype=float)
    y_train = np.asarray([example.label for example in train_examples])
    x_test = np.asarray([example.features for example in test_examples], dtype=float)

    classifier = make_classifier(random_state)
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_test)
    predicted_topk = top_k_from_probabilities(classifier.classes_, probabilities, top_k)
    predicted_labels = [row[0] for row in predicted_topk]
    metrics, rows = evaluate_recommendations(
        test_examples, predicted_labels, predicted_topk
    )
    return metrics, rows, classifier


def majority_baseline(
    train_examples: Sequence[Example],
    test_examples: Sequence[Example],
    top_k: int,
) -> tuple[dict[str, float], list[dict[str, str | float | int]]]:
    """Evaluate a majority-label baseline fitted on the training split."""
    counts = Counter(example.label for example in train_examples)
    ranked = [label for label, _ in counts.most_common()]
    predicted = ranked[0]
    topk = ranked[:top_k]
    return evaluate_recommendations(
        test_examples,
        [predicted for _ in test_examples],
        [topk for _ in test_examples],
    )


def permutation_p_value(
    train_examples: Sequence[Example],
    test_examples: Sequence[Example],
    observed_accuracy: float,
    permutations: int,
    top_k: int,
    random_state: int,
) -> float:
    """Compute a one-sided label-permutation p-value for top-1 accuracy."""
    if permutations <= 0:
        return math.nan
    rng = np.random.default_rng(random_state)
    labels = np.asarray([example.label for example in train_examples])
    exceedances = 0
    for idx in range(permutations):
        shuffled = rng.permutation(labels)
        shuffled_train = [
            Example(
                key=example.key,
                features=example.features,
                label=str(label),
                method_scores=example.method_scores,
            )
            for example, label in zip(train_examples, shuffled)
        ]
        metrics, _, _ = train_and_predict(
            shuffled_train,
            test_examples,
            top_k=top_k,
            random_state=random_state + idx + 1,
        )
        if metrics["top1_accuracy"] >= observed_accuracy:
            exceedances += 1
    return (exceedances + 1.0) / (permutations + 1.0)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write dictionaries to a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_distribution(examples: Sequence[Example]) -> str:
    """Return a compact ``label:count`` summary."""
    counts = Counter(example.label for example in examples)
    return ";".join(f"{label}:{count}" for label, count in counts.most_common())


def subset_examples(
    examples: Sequence[Example],
    study: str | None,
) -> list[Example]:
    """Filter examples by study, or return all examples when study is None."""
    if study is None:
        return list(examples)
    return [example for example in examples if example.key.study == study]


def run_one_setting(
    train_examples: Sequence[Example],
    test_examples: Sequence[Example],
    group: str,
    study: str | None,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, str | float | int]]]:
    """Run classifier, baseline, and permutation test for one setting."""
    setting_name = study or "pooled"
    train_subset = subset_examples(train_examples, study)
    test_subset = subset_examples(test_examples, study)
    if len(train_subset) < 2 or len(test_subset) < 1:
        raise ValueError(
            f"Insufficient examples for group={group}, study={setting_name}: "
            f"train={len(train_subset)}, test={len(test_subset)}"
        )

    metrics, prediction_rows, classifier = train_and_predict(
        train_subset,
        test_subset,
        top_k=args.top_k,
        random_state=args.random_state,
    )
    baseline_metrics, _ = majority_baseline(train_subset, test_subset, args.top_k)
    p_value = permutation_p_value(
        train_subset,
        test_subset,
        observed_accuracy=metrics["top1_accuracy"],
        permutations=args.permutations,
        top_k=args.top_k,
        random_state=args.random_state,
    )

    train_classes = set(example.label for example in train_subset)
    test_classes = set(example.label for example in test_subset)
    summary: dict[str, object] = {
        "group": group,
        "study": setting_name,
        "train_n": len(train_subset),
        "test_n": len(test_subset),
        "train_classes": len(train_classes),
        "test_classes": len(test_classes),
        "unseen_test_classes": len(test_classes - train_classes),
        "top1_accuracy": metrics["top1_accuracy"],
        "topk_accuracy": metrics["topk_accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mean_regret": metrics["mean_regret"],
        "median_regret": metrics["median_regret"],
        "mean_norm_regret": metrics["mean_norm_regret"],
        "permutation_p_value": p_value,
        "baseline_top1_accuracy": baseline_metrics["top1_accuracy"],
        "baseline_topk_accuracy": baseline_metrics["topk_accuracy"],
        "baseline_mean_regret": baseline_metrics["mean_regret"],
        "train_label_distribution": label_distribution(train_subset),
        "test_label_distribution": label_distribution(test_subset),
    }

    for feature, importance in zip(PAPYAN_NC_METRICS, classifier.feature_importances_):
        summary[f"importance_{feature}"] = float(importance)

    for row in prediction_rows:
        row["group"] = group
        row["study_setting"] = setting_name
    return summary, prediction_rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train on VGG13 NC metrics and test CSF prediction on ResNet18."
    )
    parser.add_argument(
        "--nc-file",
        type=Path,
        default=Path("neural_collapse_metrics/nc_metrics.csv"),
    )
    parser.add_argument("--train-scores-dir", type=Path, default=Path("scores_risk"))
    parser.add_argument(
        "--test-scores-dir",
        type=Path,
        default=Path("scores_risk_resnet18"),
    )
    parser.add_argument("--clip-dir", type=Path, default=Path("clip_scores"))
    parser.add_argument("--train-architecture", default="VGG13")
    parser.add_argument("--test-architecture", default="ResNet18")
    parser.add_argument("--train-score-backbone", default="Conv")
    parser.add_argument("--test-score-backbone", default="auto")
    parser.add_argument("--metric", default="AUGRC")
    parser.add_argument("--mcd", default="False")
    parser.add_argument("--groups", nargs="+", default=["all"])
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--studies", nargs="+", default=list(DEFAULT_STUDIES))
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Also run a pooled-study model.",
    )
    parser.add_argument(
        "--per-study",
        action="store_true",
        help="Run one model per study.",
    )
    parser.add_argument(
        "--no-filter-method-variants",
        action="store_true",
        help="Keep CSF variants containing 'global' or 'class'.",
    )
    parser.add_argument("--drop-confidence", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ood_eval_outputs/cross_arch_nc_csf"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the cross-architecture NC-to-CSF prediction analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    args = parse_args()
    if not args.pooled and not args.per_study:
        args.pooled = True
        args.per_study = True

    studies = set(args.studies)
    train_nc = read_nc_features(
        args.nc_file, args.train_architecture, PAPYAN_NC_METRICS
    )
    test_nc = read_nc_features(
        args.nc_file, args.test_architecture, PAPYAN_NC_METRICS
    )

    summaries = []
    all_predictions = []
    for group in args.groups:
        LOGGER.info("Running group=%s", group)
        train_scores = read_competitive_labels(
            scores_dir=args.train_scores_dir,
            architecture=args.train_architecture,
            backbone_label=args.train_score_backbone,
            metric=args.metric,
            mcd=args.mcd,
            sources=args.sources,
            group=group,
            clip_dir=args.clip_dir,
            filter_method_variants=not args.no_filter_method_variants,
            drop_confidence=args.drop_confidence,
        )
        test_scores = read_competitive_labels(
            scores_dir=args.test_scores_dir,
            architecture=args.test_architecture,
            backbone_label=args.test_score_backbone,
            metric=args.metric,
            mcd=args.mcd,
            sources=args.sources,
            group=group,
            clip_dir=args.clip_dir,
            filter_method_variants=not args.no_filter_method_variants,
            drop_confidence=args.drop_confidence,
        )
        train_examples = build_examples(train_nc, train_scores, studies)
        test_examples = build_examples(test_nc, test_scores, studies)

        study_settings: list[str | None] = []
        if args.pooled:
            study_settings.append(None)
        if args.per_study:
            study_settings.extend(args.studies)

        for study in study_settings:
            summary, predictions = run_one_setting(
                train_examples,
                test_examples,
                group=group,
                study=study,
                args=args,
            )
            summaries.append(summary)
            all_predictions.extend(predictions)
            LOGGER.info(
                "%s/%s: top1=%.3f, top%d=%.3f, regret=%.4f, p=%.4f",
                group,
                summary["study"],
                summary["top1_accuracy"],
                args.top_k,
                summary["topk_accuracy"],
                summary["mean_regret"],
                summary["permutation_p_value"],
            )

    summary_path = args.output_dir / "summary.csv"
    predictions_path = args.output_dir / "predictions.csv"
    write_csv(summary_path, summaries)
    write_csv(predictions_path, all_predictions)
    LOGGER.info("Wrote %s", summary_path)
    LOGGER.info("Wrote %s", predictions_path)


if __name__ == "__main__":
    main()
