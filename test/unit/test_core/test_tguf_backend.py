import numpy as np
import pytest
import torch

from tgm.core._storage import DGSliceTracker
from tgm.core._storage.backends.tguf_backend import DGStorageTGUFBackend
from tgm.data import DGData, TGUFData

tguf = pytest.importorskip('tguf')


def _write_tguf(
    path,
    num_edges=10,
    msg_dim=4,
    val_start=None,
    test_start=None,
    negatives_start_e_id=None,
    negatives_per_edge=0,
    node_feat_dim=0,
    node_feat_capacity=0,
    label_dim=0,
    label_capacity=0,
):
    schema = tguf.TGUFSchema(
        path=str(path),
        edge_capacity=num_edges,
        msg_dim=msg_dim,
        val_start=val_start,
        test_start=test_start,
        negatives_start_e_id=negatives_start_e_id,
        negatives_per_edge=negatives_per_edge,
        node_feat_capacity=node_feat_capacity,
        node_feat_dim=node_feat_dim,
        label_dim=label_dim,
        label_capacity=label_capacity,
    )
    builder = tguf.TGUFBuilder(schema)

    batch_kwargs = dict(
        src=np.arange(num_edges, dtype=np.int64),
        dst=np.arange(1, num_edges + 1, dtype=np.int64),
        time=np.arange(num_edges, dtype=np.int64) * 10,
        msg=np.random.randn(num_edges, msg_dim).astype(np.float32),
    )
    if negatives_per_edge > 0:
        n_neg_rows = num_edges - (negatives_start_e_id or 0)
        batch_kwargs['neg_dst'] = np.tile(
            np.arange(negatives_per_edge, dtype=np.int64), (n_neg_rows, 1)
        )
    builder.append_edges(tguf.Batch(**batch_kwargs))

    if node_feat_dim > 0:
        n_id = np.arange(node_feat_capacity, dtype=np.int64)
        node_feat = np.random.randn(node_feat_capacity, node_feat_dim).astype(
            np.float32
        )
        builder.append_node_feats(n_id, node_feat)

    if label_dim > 0:
        n_id = np.arange(label_capacity, dtype=np.int64) % (num_edges + 1)
        builder.append_labels(
            n_id=n_id,
            time=np.arange(label_capacity, dtype=np.int64) * 5,
            target=np.random.randn(label_capacity, label_dim).astype(np.float32),
        )

    builder.finalize()
    return path


@pytest.fixture
def tguf_path(tmp_path):
    """Path to a .tguf file with 10 edges, msg_dim=4, no splits/negatives."""
    return _write_tguf(tmp_path / 'data.tguf')


@pytest.fixture
def storage(tguf_path):
    return DGStorageTGUFBackend(TGUFData.from_tguf(str(tguf_path)))


def test_accepts_tguf_data(tguf_path):
    assert DGStorageTGUFBackend.accepts(TGUFData.from_tguf(str(tguf_path)))


def test_accepts_rejects_non_tguf_data():
    assert not DGStorageTGUFBackend.accepts(object())


def test_init_requires_tguf_data():
    edge_index = torch.IntTensor([[2, 2], [2, 4]])
    edge_timestamps = torch.LongTensor([1, 5])
    data = DGData.from_raw(edge_timestamps, edge_index)
    with pytest.raises(TypeError):
        DGStorageTGUFBackend(data)


def test_get_edges_full_range(storage):
    src, dst, time = storage.get_edges(DGSliceTracker())
    assert src.dtype == torch.int32
    assert dst.dtype == torch.int32
    assert time.dtype == torch.int64
    assert src.tolist() == list(range(10))
    assert dst.tolist() == list(range(1, 11))
    assert time.tolist() == [t * 10 for t in range(10)]


def test_get_edges_idx_slice(storage):
    src, dst, time = storage.get_edges(DGSliceTracker(start_idx=2, end_idx=5))
    assert src.tolist() == [2, 3, 4]
    assert dst.tolist() == [3, 4, 5]
    assert time.tolist() == [20, 30, 40]


def test_get_edges_empty_range(storage):
    src, dst, time = storage.get_edges(DGSliceTracker(start_idx=5, end_idx=5))
    assert src.numel() == 0
    assert dst.numel() == 0
    assert time.numel() == 0


def test_get_edges_out_of_range_clamps(storage):
    # _resolve_edge_index_range clamps start_idx/end_idx into [0, length).
    src, _, _ = storage.get_edges(DGSliceTracker(start_idx=-5, end_idx=1000))
    assert src.numel() == 10


def test_get_start_time_idx_bounds(storage):
    assert storage.get_start_time(DGSliceTracker()) == 0
    assert storage.get_start_time(DGSliceTracker(start_idx=2, end_idx=5)) == 20
    assert storage.get_start_time(DGSliceTracker(start_idx=5, end_idx=5)) is None


def test_get_end_time_idx_bounds(storage):
    assert storage.get_end_time(DGSliceTracker()) == 90
    assert storage.get_end_time(DGSliceTracker(start_idx=2, end_idx=5)) == 40
    assert storage.get_end_time(DGSliceTracker(start_idx=5, end_idx=5)) is None


def test_start_time_end_time_bounds_are_ignored(storage):
    # Current TGUF behavior: time bounds are ignored because only
    # index-based slicing is implemented.
    full = DGSliceTracker()
    narrowed_by_time = DGSliceTracker(start_time=50, end_time=60)
    assert storage.get_start_time(narrowed_by_time) == storage.get_start_time(full)
    assert storage.get_end_time(narrowed_by_time) == storage.get_end_time(full)


def test_get_num_events(storage):
    assert storage.get_num_events(DGSliceTracker()) == 10
    assert storage.get_num_events(DGSliceTracker(start_idx=2, end_idx=5)) == 3
    assert storage.get_num_events(DGSliceTracker(start_idx=5, end_idx=5)) == 0


def test_get_num_timestamps(storage):
    assert storage.get_num_timestamps(DGSliceTracker()) == 10
    assert storage.get_num_timestamps(DGSliceTracker(start_idx=2, end_idx=5)) == 3
    assert storage.get_num_timestamps(DGSliceTracker(start_idx=5, end_idx=5)) == 0


def test_get_edge_x(storage):
    edge_x = storage.get_edge_x(DGSliceTracker())
    assert edge_x.shape == (10, 4)


def test_get_edge_x_dim(storage):
    assert storage.get_edge_x_dim() == 4


def test_get_static_node_x(tmp_path):
    path = _write_tguf(tmp_path / 'data.tguf', node_feat_dim=3, node_feat_capacity=11)
    storage = DGStorageTGUFBackend(TGUFData.from_tguf(str(path)))
    node_x = storage.get_static_node_x()
    assert node_x.shape == (11, 3)
    assert storage.get_static_node_x_dim() == 3


def test_get_static_node_x_no_features(storage):
    assert storage.get_static_node_x() is None
    assert storage.get_static_node_x_dim() is None


def test_get_tguf_precomputed_negatives(tmp_path):
    path = _write_tguf(
        tmp_path / 'data.tguf',
        num_edges=10,
        negatives_start_e_id=6,
        negatives_per_edge=3,
    )
    storage = DGStorageTGUFBackend(TGUFData.from_tguf(str(path)))

    neg = storage.get_tguf_precomputed_negatives(
        DGSliceTracker(start_idx=6, end_idx=10)
    )
    assert neg.shape == (4, 3)
    assert neg.dtype == torch.int32


def test_get_tguf_precomputed_negatives_empty_range(tmp_path):
    path = _write_tguf(
        tmp_path / 'data.tguf', negatives_start_e_id=6, negatives_per_edge=3
    )
    storage = DGStorageTGUFBackend(TGUFData.from_tguf(str(path)))
    neg = storage.get_tguf_precomputed_negatives(DGSliceTracker(start_idx=6, end_idx=6))
    assert neg is None


def test_get_tguf_precomputed_negatives_raises_without_negatives(storage):
    with pytest.raises(RuntimeError):
        storage.get_tguf_precomputed_negatives(DGSliceTracker())


def test_get_node_y_no_labels(storage):
    assert storage.get_node_y(DGSliceTracker()) is None


def test_get_node_y_dim_no_labels(storage):
    assert storage.get_node_y_dim() is None


def test_get_node_y_raises_when_labels_present(tmp_path):
    path = _write_tguf(tmp_path / 'data.tguf', label_dim=2, label_capacity=2)
    storage = DGStorageTGUFBackend(TGUFData.from_tguf(str(path)))
    with pytest.raises(NotImplementedError):
        storage.get_node_y(DGSliceTracker())


@pytest.mark.parametrize(
    'method,args',
    [
        ('get_nodes', (DGSliceTracker(),)),
        ('get_node_labels', (DGSliceTracker(),)),
        ('get_nbrs', (torch.tensor([0]), 5, DGSliceTracker(), True)),
    ],
)
def test_unimplemented_methods_raise(storage, method, args):
    with pytest.raises(NotImplementedError):
        getattr(storage, method)(*args)


def test_get_node_events_always_empty(storage):
    node_ids, node_time = storage.get_node_events(DGSliceTracker())
    assert node_ids.numel() == 0
    assert node_time.numel() == 0


def test_get_node_x_always_none(storage):
    assert storage.get_node_x(DGSliceTracker()) is None


def test_get_node_x_dim_always_none(storage):
    assert storage.get_node_x_dim() is None


def test_get_node_type_always_none(storage):
    assert storage.get_node_type() is None


def test_get_edge_type_always_none(storage):
    assert storage.get_edge_type(DGSliceTracker()) is None
