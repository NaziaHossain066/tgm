from __future__ import annotations

import torch

from tgm.core import DGBatch, DGraph
from tgm.hooks.base import StatelessHook
from tgm.hooks.registry import hook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


@hook
class TGUFPrecomputedNegativeEdgeSamplerHook(StatelessHook):
    """Load pre-generated negative edges baked into a `.tguf` file.
    This reads negatives directly from the memory-mapped `.tguf` file via
    `DGStorageTGUFBackend.get_tguf_precomputed_negatives`. Only usable with
    a `DGraph` backed by `DGStorageTGUFBackend`.

    Args:
        id (str): A unique identifier for the hook.

    Attributes produced:
        neg (Tensor[int32]): Unique negative destination node ids across the batch.
        neg_batch_list (list[Tensor[int32]]): Per-edge negative candidate lists,
            aligned with `batch.edge_src`, taken directly from the file's
            baked-in `neg_dst` column.
        neg_time (Tensor[int64]): Randomly sampled timestamps for each negative,
            drawn uniformly from `[batch.edge_time.min(), batch.edge_time.max()]`
            with a fixed seed for reproducibility.
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time'}
    _cls_produces = {'neg', 'neg_batch_list', 'neg_time'}

    def __init__(self, id: str | None = None) -> None:
        super().__init__()
        self._id = id
        self.__post_init__()

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        if batch.edge_src.size(0) == 0:
            batch_neg = torch.empty(0, dtype=torch.int32, device=dg.device)
            batch_neg_time = torch.empty(0, dtype=torch.int64, device=dg.device)
            batch_neg_batch_list = []
        else:
            get_tguf_precomputed_negatives = getattr(
                dg._storage, 'get_tguf_precomputed_negatives', None
            )
            if get_tguf_precomputed_negatives is None:
                raise TypeError(
                    f'{self.__class__.__name__} requires a DGraph backed by '
                    f'DGStorageTGUFBackend, got {type(dg._storage).__name__}'
                )

            neg_dst = get_tguf_precomputed_negatives(dg._slice)
            if neg_dst is None:
                raise ValueError('No precomputed negatives found in this .tguf file')
            neg_dst = neg_dst.to(dg.device)

            batch_neg_batch_list = list(torch.unbind(neg_dst, dim=0))
            batch_neg = torch.unique(neg_dst.flatten())

            # This is a heuristic. For our fake (negative) link times,
            # we pick random time stamps within [batch.start_time, batch.end_time].
            # Using random times on the whole graph will likely produce information
            # leakage, making the prediction easier than it should be.
            
            # Use generator to local constrain rng for reproducibility
            gen = torch.Generator(device=dg.device)
            gen.manual_seed(0)
            batch_neg_time = torch.randint(
                int(batch.edge_time.min().item()),
                int(batch.edge_time.max().item()) + 1,
                (batch_neg.size(0),),
                device=dg.device,
                generator=gen,
            )

        self.add_batch_attribute(batch, 'neg', batch_neg)
        self.add_batch_attribute(batch, 'neg_batch_list', batch_neg_batch_list)
        self.add_batch_attribute(batch, 'neg_time', batch_neg_time)
        return batch
