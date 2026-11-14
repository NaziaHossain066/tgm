import numpy as np
import pytest
import torch

from tgm import TimeDeltaDG
from tgm.data import TGUFData

tguf = pytest.importorskip('tguf')


def _write_tguf(
    path, num_edges=10, msg_dim=4, val_start=None, test_start=None, label_time=None
):
    label_kwargs = {}
    if label_time is not None:
        label_kwargs = dict(label_dim=1, label_capacity=len(label_time))
    schema = tguf.TGUFSchema(
        path=str(path),
        edge_capacity=num_edges,
        msg_dim=msg_dim,
        val_start=val_start,
        test_start=test_start,
        **label_kwargs,
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
    if label_time is not None:
        builder.append_labels(
            n_id=np.arange(len(label_time), dtype=np.int64),
            time=np.array(label_time, dtype=np.int64),
            target=np.zeros((len(label_time), 1), dtype=np.float32),
        )
    builder.finalize()
    return path


@pytest.fixture
def tguf_path(tmp_path):
    """Path to a .tguf file with 10 edges and no val/test split."""
    return _write_tguf(tmp_path / 'data.tguf')


@pytest.fixture
def data(tguf_path):
    """Default TGUFData view with time_delta='r', full [0, 10) range, no splits."""
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


@pytest.fixture
def split_data(tmp_path):
    """TGUFData over 10 edges with val starting at edge 6 and test at edge 8."""
    path = _write_tguf(tmp_path / 'data.tguf', val_start=6, test_start=8)
    return TGUFData.from_tguf(str(path), time_delta='s')


def test_split_shares_store(split_data):
    for view in split_data.split():
        assert view.store is split_data.store
        assert view.events is split_data.events


def test_split_sets_offset_and_length(split_data):
    windows = [(v.offset, v.length) for v in split_data.split()]
    assert windows == [(0, 6), (6, 2), (8, 2)]


def test_split_preserves_time_delta(split_data):
    for view in split_data.split():
        assert view.time_delta == TimeDeltaDG('s')


def test_split_of_split_stays_in_window(split_data):
    """Splitting a view intersects the store's splits with the view's window."""
    _, val, _ = split_data.split()
    windows = [(v.offset, v.length) for v in val.split()]
    assert windows == [(6, 0), (6, 2), (8, 0)]


def test_from_tguf_with_val_test_start(tmp_path):
    path = _write_tguf(tmp_path / 'data.tguf', val_start=6, test_start=8)
    data = TGUFData.from_tguf(str(path))
    assert data.store.train_split.start == 0
    assert data.store.train_split.end == 6
    assert data.store.val_split.start == 6
    assert data.store.val_split.end == 8
    assert data.store.test_split.start == 8
    assert data.store.test_split.end == 10


def test_label_events_empty_without_labels(data):
    assert len(data.events) == 0


def test_label_events_from_file(tmp_path):
    # Edge times 0, 10, ..., 90. Labels at 5, 25, 25, 55 form 3 events.
    path = _write_tguf(tmp_path / 'data.tguf', label_time=[5, 25, 25, 55])
    events = TGUFData.from_tguf(str(path)).events
    assert events.time.tolist() == [5, 25, 55]
    assert events.row_offset.tolist() == [0, 1, 3, 4]
    assert events.cutoff.tolist() == [1, 3, 6]


@pytest.mark.parametrize(
    'label_time',
    [
        pytest.param([25, 5, 25], id='unsorted'),
        pytest.param([25, 5, 5], id='decreasing'),
    ],
)
def test_from_tguf_rejects_unsorted_label_times(tmp_path, label_time):
    # The builder does not check label order, so TGUFData has to.
    path = _write_tguf(tmp_path / 'data.tguf', label_time=label_time)
    with pytest.raises(ValueError, match='strictly ascending'):
        TGUFData.from_tguf(str(path))


def test_window_cutoff_clamps_to_window(tmp_path):
    # Edge times 0, 10, ..., 90, so the cutoffs are 1, 3, 6, 9.
    path = _write_tguf(tmp_path / 'data.tguf', label_time=[5, 25, 55, 85])
    events = TGUFData.from_tguf(str(path)).events
    assert events.cutoff.tolist() == [1, 3, 6, 9]
    assert events.window_cutoff(0, 4, 3, 6).tolist() == [3, 3, 6, 6]
    assert events.window_cutoff(1, 3, 0, 10).tolist() == [3, 6]
