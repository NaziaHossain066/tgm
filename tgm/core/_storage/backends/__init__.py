from tgm.core._storage.backends.array_backend import DGStorageArrayBackend
from tgm.core._storage.backends.tguf_backend import DGStorageTGUFBackend

DGStorageBackends = {
    'ArrayBackend': DGStorageArrayBackend,
    'TGUFBackend': DGStorageTGUFBackend,
}

DGStorage = DGStorageArrayBackend
