from typing import Any, Optional, Set, Tuple

import torch
from torch import Tensor

from ..base import DGSliceTracker, DGStorageBase


class DGStorageTGUFBackend(DGStorageBase):
    """TGUF backed implementation of temporal graph storage engine.

    Under active development, methods are being implemented incrementally.
    Unimplemented methods raise `NotImplementedError`.
    """

    @classmethod
    def accepts(cls, data: Any) -> bool:
        from tgm.data import TGUFData  # Avoid circular dependency

        return isinstance(data, TGUFData)

    def __init__(self, data: 'TGUFData') -> None:  # type: ignore
        if not self.accepts(data):
            raise TypeError(f'DGStorageTGUFBackend requires TGUFData, got {type(data)}')
        self._data = data

    def get_start_time(self, slice: DGSliceTracker) -> Optional[int]:
        lb, ub = self._resolve_edge_index_range(slice)
        if lb >= ub:
            return None
        ts = self._data.store.gather_timestamps(torch.tensor([lb], dtype=torch.int64))
        return int(ts[0])

    def get_end_time(self, slice: DGSliceTracker) -> Optional[int]:
        lb, ub = self._resolve_edge_index_range(slice)
        if lb >= ub:
            return None
        ts = self._data.store.gather_timestamps(torch.tensor([ub - 1], dtype=torch.int64))
        return int(ts[0])

    def get_nodes(self, slice: DGSliceTracker) -> Set[int]:
        raise NotImplementedError

    def get_edges(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor, Tensor]:
        # TGUF stores node IDs as int64, whereas tgm consistently uses int32
        # (e.g., DGData and PADDED_NODE_ID). Downcast here so downstream code
        # behaves identically regardless of the storage backend.
        lb, ub = self._resolve_edge_index_range(slice)
        if lb >= ub:
            return (
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int64),
            )
        batch = self._data.store.get_batch(lb, ub - lb)
        src = torch.as_tensor(batch.src).to(torch.int32)
        dst = torch.as_tensor(batch.dst).to(torch.int32)
        time = torch.as_tensor(batch.time).to(torch.int64)
        return src, dst, time

    def get_node_events(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor]:
        # TGUF has no concept of dynamic (time-varying) node features distinct
        # from static node features or labels, always returns empty tensors.
        return torch.empty(0, dtype=torch.int), torch.empty(0, dtype=torch.long)

    def get_node_labels(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def get_num_timestamps(self, slice: DGSliceTracker) -> int:
        lb, ub = self._resolve_edge_index_range(slice)
        if lb >= ub:
            return 0
        batch = self._data.store.get_batch(lb, ub - lb)
        return len(batch.time.unique())

    def get_num_events(self, slice: DGSliceTracker) -> int:
        lb, ub = self._resolve_edge_index_range(slice)
        return max(0, ub - lb)

    def get_node_x(self, slice: DGSliceTracker) -> Optional[Tensor]:
        # TGUF has no concept of dynamic (time-varying) node features.
        return None

    def get_node_y(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._data.store.label_dim == 0:
            return None
        raise NotImplementedError(
            'Label reading not yet implemented for DGStorageTGUFBackend.'
        )

    def get_edge_x(self, slice: DGSliceTracker) -> Optional[Tensor]:
        store = self._data.store
        if store.msg_dim == 0:
            return None
        lb, ub = self._resolve_edge_index_range(slice)
        if lb >= ub:
            return None
        e_id = torch.arange(lb, ub, dtype=torch.int64)
        return store.gather_msgs(e_id)

    def get_edge_type(self, slice: DGSliceTracker) -> Optional[Tensor]:
        # TGUFSchema has no edge-type field, heterogeneous/knowledge graphs unsupported.
        return None

    def get_tguf_precomputed_negatives(self, slice: DGSliceTracker) -> Optional[Tensor]:
        """Return precomputed negative destinations for edges in this slice.

        tguf-specific extension, not part of DGStorageBase's abstract interface.
        Raises (from tguf itself) if the .tguf file has no precomputed negatives baked in.

        Shape: (num_edges_in_slice, negatives_per_edge).
        """
        import tguf

        lb, ub = self._resolve_edge_index_range(slice)
        if lb >= ub:
            return None
        batch = self._data.store.get_batch(
            lb, ub - lb, strategy=tguf.NegStrategy.PreComputed
        )
        return torch.as_tensor(batch.neg_dst).to(torch.int32)

    def get_static_node_x(self) -> Optional[Tensor]:
        store = self._data.store
        if store.node_feat_dim == 0:
            return None
        n_id = torch.arange(store.node_count, dtype=torch.int64)
        return store.gather_node_feats(n_id)

    def get_node_type(self) -> Optional[Tensor]:
        # TGUFSchema has no node-type field, heterogeneous/knowledge graphs unsupported.
        return None

    def get_node_x_dim(self) -> Optional[int]:
        # TGUF has no concept of dynamic (time-varying) node features, see get_node_events.
        return None

    def get_node_y_dim(self) -> Optional[int]:
        dim = self._data.store.label_dim
        return dim if dim > 0 else None

    def get_edge_x_dim(self) -> Optional[int]:
        dim = self._data.store.msg_dim
        return dim if dim > 0 else None

    def get_static_node_x_dim(self) -> Optional[int]:
        dim = self._data.store.node_feat_dim
        return dim if dim > 0 else None

    def get_nbrs(
        self,
        seed_nodes: Tensor,
        num_nbrs: int,
        slice: DGSliceTracker,
        directed: bool,
    ) -> Tuple[Tensor, ...]:
        raise NotImplementedError

    def _resolve_edge_index_range(self, slice: DGSliceTracker) -> Tuple[int, int]:
        """Resolve a slice to a global half-open edge-index range."""

        #Only index-based slicing is currently supported. Time bounds
        #(start_time/end_time) are ignored because TGStore does not
        #provide a time-to-index lookup.
        
        offset = self._data.offset
        length = self._data.length
        local_lb = slice.start_idx if slice.start_idx is not None else 0
        local_ub = slice.end_idx if slice.end_idx is not None else length
        local_lb = max(0, min(local_lb, length))
        local_ub = max(0, min(local_ub, length))
        return offset + local_lb, offset + local_ub
