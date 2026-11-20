from typing import Type
from tgm.core._storage.base import DGStorageBase
from tgm.core._storage.backends.array_backend import DGStorageArrayBackend
from tgm.core._storage.backends.tguf_backend import DGStorageTGUFBackend

DGStorageBackends = {
    'ArrayBackend': DGStorageArrayBackend,
    'TGUFBackend': DGStorageTGUFBackend,
}

DGStorage: Type[DGStorageBase] = DGStorageArrayBackend
