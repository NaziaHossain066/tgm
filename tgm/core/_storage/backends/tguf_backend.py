from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import torch
from torch import Tensor

from ..base import DGSliceTracker, DGStorageBase

# (e_lo, e_hi, r_lo, r_hi): window-local edge rows and label rows of a slice.
_Resolved = Tuple[int, int, int, int]


class DGStorageTGUFBackend(DGStorageBase):
    """TGUF backed implementation of temporal graph storage engine.

    Edges and label rows form one timeline ordered by time, with labels before
    edges at the same timestamp (TGStore's edge cutoff rule). Event indices are
    positions on that timeline. The rows of one label event are contiguous on it,
    so one entry per label event is enough to split any position into edge rows
    and label rows.
    """

    _TIME_CACHE_SIZE = 1024

    @classmethod
    def accepts(cls, data: Any) -> bool:
        from tgm.data import TGUFData  # Avoid circular dependency

        return isinstance(data, TGUFData)

    def __init__(self, data: 'TGUFData') -> None:  # type: ignore
        if not self.accepts(data):
            raise TypeError(f'DGStorageTGUFBackend requires TGUFData, got {type(data)}')
        if data.store.node_count > torch.iinfo(torch.int32).max:
            raise ValueError('Node ids exceed the int32 range used by TGM')

        self._data = data
        self._store = data.store
        self._e0, self._E = data.offset, data.length

        # Per label event in the window: row offsets (window-local, K_w + 1 entries),
        # timestamps, and the position of each event's first row on the timeline.
        # Cutoffs outside the edge window place the event at the window's start or end.
        events, ls, le = data.events, data.label_start, data.label_end
        rows = events.row_offset[ls : le + 1].numpy()
        self._r0 = int(rows[0])
        self._ev_row = rows - self._r0
        self._ev_time = events.time[ls:le].numpy()
        cutoff = events.window_cutoff(ls, le, self._e0, self._e0 + self._E).numpy()
        self._ev_pos = self._ev_row[:-1] + (cutoff - self._e0)
        self._R = int(self._ev_row[-1])

        self._time_cache: Dict[Tuple[int, str], Tuple[int, int]] = {}

    def get_start_time(self, slice: DGSliceTracker) -> Optional[int]:
        e_lo, e_hi, r_lo, r_hi = self._resolve(slice)
        times = []
        if e_lo < e_hi:
            times.append(int(self._edges(e_lo, e_lo + 1).time[0]))
        if r_lo < r_hi:
            times.append(int(self._ev_time[self._event_of_row(r_lo)]))
        return min(times) if times else None

    def get_end_time(self, slice: DGSliceTracker) -> Optional[int]:
        e_lo, e_hi, r_lo, r_hi = self._resolve(slice)
        times = []
        if e_lo < e_hi:
            times.append(int(self._edges(e_hi - 1, e_hi).time[0]))
        if r_lo < r_hi:
            times.append(int(self._ev_time[self._event_of_row(r_hi - 1)]))
        return max(times) if times else None

    def get_nodes(self, slice: DGSliceTracker) -> Set[int]:
        # Like the Array backend, nodes come from edges only, not from labels.
        e_lo, e_hi, _, _ = self._resolve(slice)
        if e_lo >= e_hi:
            return set()
        batch = self._edges(e_lo, e_hi)
        src, dst = torch.as_tensor(batch.src), torch.as_tensor(batch.dst)
        return set(torch.cat([src, dst]).unique().tolist())

    def get_edges(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor, Tensor]:
        # TGUF stores node IDs as int64, whereas tgm consistently uses int32
        # (e.g., DGData and PADDED_NODE_ID). Downcast here so downstream code
        # behaves identically regardless of the storage backend.
        e_lo, e_hi, _, _ = self._resolve(slice)
        if e_lo >= e_hi:
            return (
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int64),
            )
        batch = self._edges(e_lo, e_hi)
        src = torch.as_tensor(batch.src).to(torch.int32)
        dst = torch.as_tensor(batch.dst).to(torch.int32)
        # batch.time is a view into the read-only memory map, copy it.
        time = torch.as_tensor(batch.time).clone()
        return src, dst, time

    def get_node_events(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor]:
        # TGUF has no concept of dynamic (time-varying) node features distinct
        # from static node features or labels, always returns empty tensors.
        return torch.empty(0, dtype=torch.int), torch.empty(0, dtype=torch.long)

    def get_node_labels(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor]:
        _, _, r_lo, r_hi = self._resolve(slice)
        if r_lo >= r_hi:
            return torch.empty(0, dtype=torch.int), torch.empty(0, dtype=torch.long)
        labels = self._labels(r_lo, r_hi)
        return torch.as_tensor(labels.n_id).to(torch.int32), torch.as_tensor(
            labels.time
        ).clone()

    def get_num_timestamps(self, slice: DGSliceTracker) -> int:
        e_lo, e_hi, r_lo, r_hi = self._resolve(slice)
        times = []
        if e_lo < e_hi:
            times.append(torch.as_tensor(self._edges(e_lo, e_hi).time))
        if r_lo < r_hi:
            k0, k1 = self._event_of_row(r_lo), self._event_of_row(r_hi - 1) + 1
            times.append(torch.from_numpy(self._ev_time[k0:k1]))
        return int(torch.cat(times).unique().numel()) if times else 0

    def get_num_events(self, slice: DGSliceTracker) -> int:
        e_lo, e_hi, r_lo, r_hi = self._resolve(slice)
        return (e_hi - e_lo) + (r_hi - r_lo)

    def get_node_x(self, slice: DGSliceTracker) -> Optional[Tensor]:
        # TGUF has no concept of dynamic (time-varying) node features.
        return None

    def get_node_y(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._store.label_dim == 0:
            return None
        e_lo, e_hi, r_lo, r_hi = self._resolve(slice)
        if r_lo >= r_hi:
            return None

        labels = self._labels(r_lo, r_hi)
        n_id = torch.as_tensor(labels.n_id)
        time = torch.as_tensor(labels.time)
        target = torch.as_tensor(labels.target)

        # Size the sparse tensor like the Array backend: max node id over the
        # slice's labels and edges, max time from the slice bound if one is set.
        max_node_id = int(n_id.max())
        if e_lo < e_hi:
            batch = self._edges(e_lo, e_hi)
            max_node_id = max(
                max_node_id,
                int(torch.as_tensor(batch.src).max()),
                int(torch.as_tensor(batch.dst).max()),
            )
        max_time = slice.end_time or self.get_end_time(slice)
        shape = (max_time + 1, max_node_id + 1, self._store.label_dim)  # type: ignore
        return torch.sparse_coo_tensor(torch.stack([time, n_id]), target, shape)

    def get_edge_x(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._store.msg_dim == 0:
            return None
        e_lo, e_hi, _, _ = self._resolve(slice)
        if e_lo >= e_hi:
            return None
        return torch.as_tensor(self._edges(e_lo, e_hi).msg).clone()

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

        e_lo, e_hi, _, _ = self._resolve(slice)
        if e_lo >= e_hi:
            return None
        batch = self._edges(e_lo, e_hi, strategy=tguf.NegStrategy.PreComputed)
        return torch.as_tensor(batch.neg_dst).to(torch.int32)

    def get_static_node_x(self) -> Optional[Tensor]:
        store = self._store
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
        # Like a DGData split without labels, a window without label rows has none.
        dim = self._store.label_dim
        return dim if dim > 0 and self._R > 0 else None

    def get_edge_x_dim(self) -> Optional[int]:
        dim = self._store.msg_dim
        return dim if dim > 0 else None

    def get_static_node_x_dim(self) -> Optional[int]:
        dim = self._store.node_feat_dim
        return dim if dim > 0 else None

    def get_nbrs(
        self,
        seed_nodes: Tensor,
        num_nbrs: int,
        slice: DGSliceTracker,
        directed: bool,
    ) -> Tuple[Tensor, ...]:
        raise NotImplementedError

    def _resolve(self, slice: DGSliceTracker) -> _Resolved:
        """Resolve a slice to window-local edge rows [e_lo, e_hi) and label rows [r_lo, r_hi)."""
        e_lo, r_lo, e_hi, r_hi = 0, 0, self._E, self._R

        # Time bounds (end_time is inclusive). Each stream is searched on its own;
        # a time bound never splits a label event.
        if slice.start_time is not None:
            e_lo, r_lo = self._time_bound(slice.start_time, 'left')
        if slice.end_time is not None:
            e_hi, r_hi = self._time_bound(slice.end_time, 'right')

        # Index bounds are positions on the combined timeline, intersected with
        # the time range and split back into edge rows and label rows.
        if slice.start_idx is not None or slice.end_idx is not None:
            pos_lo, pos_hi = e_lo + r_lo, e_hi + r_hi
            if slice.start_idx is not None:
                pos_lo = max(pos_lo, slice.start_idx)
            if slice.end_idx is not None:
                pos_hi = min(pos_hi, slice.end_idx)
            pos_lo = max(0, min(pos_lo, self._E + self._R))
            pos_hi = max(pos_lo, min(pos_hi, self._E + self._R))
            r_lo, r_hi = self._labels_before_pos(pos_lo), self._labels_before_pos(pos_hi)
            e_lo, e_hi = pos_lo - r_lo, pos_hi - r_hi

        return e_lo, max(e_lo, e_hi), r_lo, max(r_lo, r_hi)

    def _time_bound(self, t: int, side: str) -> Tuple[int, int]:
        """(edges, label rows) with time < t (side='left') or <= t (side='right')."""
        key = (t, side)
        if key not in self._time_cache:
            if len(self._time_cache) >= self._TIME_CACHE_SIZE:
                self._time_cache.clear()
            edge_ts = torch.as_tensor(self._store.get_batch(self._e0, self._E).time)
            e = int(torch.searchsorted(edge_ts, torch.tensor(t), side=side))
            r = int(self._ev_row[np.searchsorted(self._ev_time, t, side=side)])
            self._time_cache[key] = (e, r)
        return self._time_cache[key]

    def _labels_before_pos(self, pos: int) -> int:
        """Number of window label rows at combined-timeline positions < pos."""
        j = int(np.searchsorted(self._ev_pos, pos)) - 1
        if j < 0:
            return 0
        size = int(self._ev_row[j + 1] - self._ev_row[j])
        return int(self._ev_row[j]) + min(size, pos - int(self._ev_pos[j]))

    def _event_of_row(self, r: int) -> int:
        """Window label event that contains window label row r."""
        return int(np.searchsorted(self._ev_row, r, side='right')) - 1

    def _edges(self, e_lo: int, e_hi: int, strategy: Any = None) -> Any:
        if strategy is None:
            return self._store.get_batch(self._e0 + e_lo, e_hi - e_lo)
        return self._store.get_batch(self._e0 + e_lo, e_hi - e_lo, strategy=strategy)

    def _labels(self, r_lo: int, r_hi: int) -> Any:
        return self._store.get_label_batch(self._r0 + r_lo, r_hi - r_lo)
