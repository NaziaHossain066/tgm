from __future__ import annotations

from typing import Optional

from tgm.core import TimeDeltaDG


class TGUFData:
    """A lazy handle over a memory-mapped `.tguf` file, used by `DGStorageTGUFBackend`.
    
    Defers all reads to the wrapped `tguf.TGStore`, loads data lazily as it's queried.
    Holds no materialized tensors and never loads the full graph into RAM. 
    
    """

    def __init__(
        self,
        store: tguf.TGStore, # type: ignore
        time_delta: TimeDeltaDG | str,
        offset: int = 0,
        length: Optional[int] = None,
    ) -> None:
        self.store = store
        if isinstance(time_delta, str):
            time_delta = TimeDeltaDG(time_delta)
        self.time_delta = time_delta
        self.offset = offset
        self.length = store.edge_count - offset if length is None else length

    @property
    def num_nodes(self) -> int:
        return self.store.node_count

    def slice(self, start: int, end: int) -> 'TGUFData':
        """Return a new TGUFData sharing the same store, windowed to local [start, end).

        Does not reopen the file or copy any data, shares `self.store` with the
        parent, only the offset/length window differs. Used to build independent
        train/val/test views over the same underlying `.tguf` file.
        """
        if start < 0 or end > self.length or start > end:
            raise ValueError(
                f'Invalid slice [{start}:{end}) for TGUFData of length {self.length}'
            )
        return TGUFData(
            self.store, self.time_delta, offset=self.offset + start, length=end - start
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
