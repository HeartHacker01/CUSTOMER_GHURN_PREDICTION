# Patch sklearn to bypass the blocked _datasets_pair DLL
# This makes NearestNeighbors unavailable but LR and RF still work

import sys, types

# Create minimal stubs that satisfy the import chain
class _ArgKminStub:
    pass

class _RadiusNeighborsStub:
    pass

# Build stubs for all Cython modules in the reduction chain
for mod_name in [
    'sklearn.metrics._pairwise_distances_reduction._datasets_pair',
    'sklearn.metrics._pairwise_distances_reduction._base',
    'sklearn.metrics._pairwise_distances_reduction._argkmin',
    'sklearn.metrics._pairwise_distances_reduction._argkmin_classmode',
    'sklearn.metrics._pairwise_distances_reduction._middle_term_computer',
    'sklearn.metrics._pairwise_distances_reduction._radius_neighbors',
    'sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode',
]:
    m = types.ModuleType(mod_name)
    m.ArgKmin32 = _ArgKminStub
    m.ArgKmin64 = _ArgKminStub
    m.ArgKminClassMode32 = _ArgKminStub
    m.ArgKminClassMode64 = _ArgKminStub
    m.RadiusNeighbors32 = _RadiusNeighborsStub
    m.RadiusNeighbors64 = _RadiusNeighborsStub
    m.RadiusNeighborsClassMode32 = _RadiusNeighborsStub
    m.RadiusNeighborsClassMode64 = _RadiusNeighborsStub
    m._sqeuclidean_row_norms32 = lambda x, y: None
    m._sqeuclidean_row_norms64 = lambda x, y: None
    sys.modules[mod_name] = m
