"""Parquet-native image loading for the x9 pipeline (the ONE loader).

Handles both schemas in play:
  - HF ImageNet shards: column `image` = struct {bytes, path}, `label` int
    (train-*.parquet, validation-*.parquet)
  - x9-written shards: column `image_bytes` = binary, `label` int
    (superset-*.parquet from build_superset.py, <set>-*.parquet from
    repack_ood.py)

`ParquetImageDataset` is an IterableDataset: shards are assigned to
DataLoader workers round-robin, rows stream via pyarrow batches, JPEG bytes
decode with PIL, and the per-model timm transform applies. Row order within
a shard is deterministic, and each item carries a uid "<shard>:<row>" so
features align with labels/manifests regardless of worker interleaving.
Use num_workers <= number of shards (extra workers idle).

`row_filter` enables the fit/selection/G3 draws: a set of uids to keep
(seeded selections over the superset manifest), applied while streaming.
"""
from __future__ import annotations

import io
import pathlib
from collections.abc import Iterator

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def model_transform(model):
    """The exact eval transform for a timm model instance (per-checkpoint
    mean/std/interpolation/crop_pct; the classic silent killer if wrong)."""
    import timm
    cfg = timm.data.resolve_model_data_config(model)
    return timm.data.create_transform(**cfg, is_training=False)


class ParquetImageDataset(IterableDataset):
    def __init__(self, shards: list[str | pathlib.Path], transform,
                 row_filter: set[str] | None = None,
                 batch_size: int = 256):
        self.shards = [str(s) for s in shards]
        self.transform = transform
        self.row_filter = row_filter
        self.batch_size = batch_size

    def _columns(self, pf: pq.ParquetFile) -> tuple[str, bool]:
        names = pf.schema_arrow.names
        if "image_bytes" in names:
            return "image_bytes", False
        if "image" in names:
            return "image", True
        raise ValueError(f"no image column in {names}")

    def _iter_shard(self, shard: str) -> Iterator[tuple[torch.Tensor, int, str]]:
        name = pathlib.Path(shard).name
        pf = pq.ParquetFile(shard)
        col, is_struct = self._columns(pf)
        row = 0
        for batch in pf.iter_batches(columns=[col, "label"],
                                     batch_size=self.batch_size):
            imgs = batch.column(col).to_pylist()
            labels = batch.column("label").to_pylist()
            for img, y in zip(imgs, labels):
                uid = f"{name}:{row}"
                row += 1
                if self.row_filter is not None and uid not in self.row_filter:
                    continue
                data = img["bytes"] if is_struct else img
                pic = Image.open(io.BytesIO(data)).convert("RGB")
                yield self.transform(pic), int(y), uid

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int, str]]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        for i, shard in enumerate(self.shards):
            if i % nw == wid:
                yield from self._iter_shard(shard)


def make_loader(shards: list[str | pathlib.Path], transform,
                row_filter: set[str] | None = None, batch_size: int = 256,
                num_workers: int = 8) -> DataLoader:
    nw = min(num_workers, len(list(shards)))
    ds = ParquetImageDataset(shards, transform, row_filter=row_filter)
    return DataLoader(ds, batch_size=batch_size, num_workers=nw,
                      pin_memory=True)
