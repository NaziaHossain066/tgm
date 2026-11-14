from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from tgm.core import TimeDeltaDG


class _LabelEvents:
    """Helper table used to calculate the label rows in an event batch.

    Event batches need a fixed number of events (edges plus labels), but TGUF keeps
    edges and labels on separate timelines. This table maps a combined event range
    to its edge and label rows. Read once from the store's label event table and
    shared by all windows.

    Args:
        time (Tensor): Timestamp of each label event, ascending.
            Shape `(num_label_events,)`.
        row_offset (Tensor): Start row of each label event, plus the total label
            row count. Shape `(num_label_events + 1,)`.
        cutoff (Tensor): Number of edges before each label event.
            Shape `(num_label_events,)`.

    Note:
        A label event is the group of label rows sharing one timestamp. Event `k`
        covers label rows [row_offset[k], row_offset[k + 1]) and starts at position
        `row_offset[k] + cutoff[k]` on the combined timeline.

    """

    def __init__(self, time: Tensor, row_offset: Tensor, cutoff: Tensor) -> None:
        k = len(time)
        if len(cutoff) != k or len(row_offset) != k + 1:
            raise ValueError(
                f'Label event table shape mismatch: {k} times, {len(cutoff)} cutoffs, '
                f'{len(row_offset)} row offsets (expected {k + 1})'
            )
        if int(row_offset[0]) != 0:
            raise ValueError('Label event row offsets must start at 0')
        if not bool((row_offset.diff() > 0).all()):
            raise ValueError('Every label event must have at least one row')
        if not bool((time.diff() > 0).all()):
            raise ValueError('Label event times must be strictly ascending')
        if not bool((cutoff.diff() >= 0).all()):
            raise ValueError('Label event cutoffs must be non-decreasing')
        self.time = time
        self.row_offset = row_offset
        self.cutoff = cutoff

    @classmethod
    def from_store(cls, store: 'tguf.TGStore') -> '_LabelEvents':  # type: ignore
        if store.label_count == 0:
            empty = torch.empty(0, dtype=torch.int64)
            return cls(empty, torch.zeros(1, dtype=torch.int64), empty)
        table = store.label_event_table
        events = cls(table.time, table.row_offset, table.cutoff)
        if int(events.row_offset[-1]) != store.label_count:
            raise ValueError(
                f'Label event table covers {int(events.row_offset[-1])} rows, '
                f'but the store has {store.label_count}'
            )
        return events

    def __len__(self) -> int:
        return len(self.time)

    def window_cutoff(self, label_start: int, label_end: int, lo: int, hi: int) -> Tensor:
        """Cutoffs of events [label_start, label_end), clamped to the edge window [lo, hi].

        An event whose cutoff lies outside the window is placed at the window's first
        or last edge. The storage backend relies on this.
        """
        return self.cutoff[label_start:label_end].clamp(lo, hi)


class TGUFData:
    """A lazy handle over a memory-mapped `.tguf` file, used by `DGStorageTGUFBackend`.

    Defers all reads to the wrapped `tguf.TGStore`, loads data lazily as it's queried.
    Holds no materialized tensors and never loads the full graph into RAM.

    A view covers the edge range [offset, offset + length) and the label event
    range [label_start, label_end) of the underlying store.

    Note:
        `label_start` and `label_end` index tguf label events (groups of label rows
        sharing a timestamp), not label rows. Event batching still counts each label
        row as one event, see `_LabelEvents`.
    """

    def __init__(
        self,
        store: tguf.TGStore,  # type: ignore
        time_delta: TimeDeltaDG | str,
        offset: int = 0,
        length: Optional[int] = None,
        label_start: int = 0,
        label_end: Optional[int] = None,
        events: Optional[_LabelEvents] = None,
    ) -> None:
        self.store = store
        if isinstance(time_delta, str):
            time_delta = TimeDeltaDG(time_delta)
        self.time_delta = time_delta
        self.offset = offset
        self.length = store.edge_count - offset if length is None else length
        self.events = events if events is not None else _LabelEvents.from_store(store)
        self.label_start = label_start
        self.label_end = len(self.events) if label_end is None else label_end

    @property
    def num_nodes(self) -> int:
        return self.store.node_count

    def split(self) -> Tuple['TGUFData', 'TGUFData', 'TGUFData']:
        """Return train/val/test views using the store's edge splits.

        Views share `self.store` and copy no data, only the window differs.

        Edge ranges come from the store's edge splits, intersected with this view's
        window. Pass `val_start` / `test_start` to `from_tguf` to choose other edge
        ranges. Label events are assigned like `TGBSplit` on the Array backend: a
        split whose edges span times [t_first, t_last] keeps the label events with
        `t_first - 1 <= time < t_last`. A TGB label at time t is answered by the
        edges after t, so it belongs to the split where those edges are (e.g.
        tgbn-trade's 2009 labels belong to validation, which starts at 2010).
        """
        store = self.store
        edge_splits = (store.train_split, store.val_split, store.test_split)

        views = []
        for es in edge_splits:
            e0 = max(es.start, self.offset)
            e1 = max(min(es.end, self.offset + self.length), e0)
            l0, l1 = self._split_label_range(e0, e1)
            views.append(self._view(e0, e1 - e0, l0, l1))
        return views[0], views[1], views[2]

    def _split_label_range(self, e0: int, e1: int) -> Tuple[int, int]:
        """Label events of this view for a split with global edge rows [e0, e1)."""
        if e0 >= e1 or self.label_start >= self.label_end:
            # Like TGBSplit, a split without edges gets no labels.
            return self.label_start, self.label_start
        t_first, t_last = self.store.gather_timestamps(
            torch.tensor([e0, e1 - 1], dtype=torch.int64)
        ).tolist()
        time = self.events.time[self.label_start : self.label_end]
        l0 = int(torch.searchsorted(time, torch.tensor(t_first - 1)))
        l1 = int(torch.searchsorted(time, torch.tensor(t_last)))
        return self.label_start + l0, self.label_start + max(l1, l0)

    def _view(
        self, offset: int, length: int, label_start: int, label_end: int
    ) -> 'TGUFData':
        return TGUFData(
            self.store,
            self.time_delta,
            offset=offset,
            length=length,
            label_start=label_start,
            label_end=label_end,
            events=self.events,
        )

    @classmethod
    def from_tguf(
        cls,
        path: str,
        time_delta: TimeDeltaDG | str = 'r',
        val_start: Optional[int] = None,
        test_start: Optional[int] = None,
    ) -> 'TGUFData':
        """Open a `.tguf` file via mmap. Does not load graph data into memory.

        Args:
            path: Path to the `.tguf` file.
            time_delta: Time granularity for this graph view.
            val_start: Optional edge index marking the start of the validation split.
            test_start: Optional edge index marking the start of the test split.
        """
        import tguf

        store = tguf.TGStore.from_tguf(path, val_start, test_start)
        return cls(store, time_delta)
