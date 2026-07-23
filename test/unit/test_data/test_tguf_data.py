import numpy as np
import pytest
import torch

from tgm import TimeDeltaDG
from tgm.data import TGUFData

tguf = pytest.importorskip('tguf')


def _write_tguf(path, num_edges=10, msg_dim=4, val_start=None, test_start=None):
    schema = tguf.TGUFSchema(
        path=str(path),
        edge_capacity=num_edges,
        msg_dim=msg_dim,
        val_start=val_start,
        test_start=test_start,
    )
    builder = tguf.TGUFBuilder(schema)
    builder.append_edges(
        tguf.Batch(
            src=np.arange(num_edges, dtype=np.int64),
            dst=np.arange(1, num_edges + 1, dtype=np.int64),
            time=np.arange(num_edges, dtype=np.int64) * 10,
            msg=np.random.randn(num_edges, msg_dim).astype(np.float32),
        )
    )
    builder.finalize()
    return path


@pytest.fixture
def tguf_path(tmp_path):
    """Path to a .tguf file with 10 edges and no val/test split."""
    return _write_tguf(tmp_path / 'data.tguf')


@pytest.fixture
def data(tguf_path):
    """Default TGUFData view: time_delta='r', full [0, 10) range, no splits."""
    return TGUFData.from_tguf(str(tguf_path))


@pytest.mark.parametrize(
    'time_delta_arg,expected',
    [
        pytest.param(None, TimeDeltaDG('r'), id='default'),
        pytest.param('s', TimeDeltaDG('s'), id='string'),
        pytest.param(TimeDeltaDG('s'), TimeDeltaDG('s'), id='object'),
    ],
)
def test_from_tguf_time_delta(tguf_path, time_delta_arg, expected):
    kwargs = {} if time_delta_arg is None else {'time_delta': time_delta_arg}
    data = TGUFData.from_tguf(str(tguf_path), **kwargs)
    assert data.time_delta == expected


def test_from_tguf_no_materialized_tensors(data):
    for value in vars(data).values():
        assert not isinstance(value, torch.Tensor)


def test_from_tguf_num_nodes(data):
    assert data.num_nodes == data.store.node_count


def test_from_tguf_default_offset_and_length(data):
    assert data.offset == 0
    assert data.length == data.store.edge_count == 10


def test_slice_shares_store(data):
    sliced = data.slice(2, 6)
    assert sliced.store is data.store


def test_slice_sets_offset_and_length(data):
    sliced = data.slice(2, 6)
    assert sliced.offset == 2
    assert sliced.length == 4


def test_slice_composes(data):
    """Slicing a slice should compose offsets, not reset relative to the parent."""
    sliced_once = data.slice(2, 8)  
    sliced_twice = sliced_once.slice(1, 3)  
    assert sliced_twice.offset == 3
    assert sliced_twice.length == 2


def test_slice_preserves_time_delta(tguf_path):
    data = TGUFData.from_tguf(str(tguf_path), time_delta='s')
    sliced = data.slice(2, 6)
    assert sliced.time_delta == TimeDeltaDG('s')


@pytest.mark.parametrize(
    'start,end',
    [
        (-1, 5),  # negative start
        (0, 11),  # end beyond length
        (6, 2),  # start > end
    ],
)
def test_slice_invalid_range_raises(data, start, end):
    with pytest.raises(ValueError):
        data.slice(start, end)


def test_from_tguf_with_val_test_start(tmp_path):
    path = _write_tguf(tmp_path / 'data.tguf', val_start=6, test_start=8)
    data = TGUFData.from_tguf(str(path))
    assert data.store.train_split.start == 0
    assert data.store.train_split.end == 6
    assert data.store.val_split.start == 6
    assert data.store.val_split.end == 8
    assert data.store.test_split.start == 8
    assert data.store.test_split.end == 10
