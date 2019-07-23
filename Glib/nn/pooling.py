from torch_geometric.utils import scatter_
from torch_geometric.utils import add_remaining_self_loops

def graph_max_pool(x, edge_index):
    edge_index, _ = add_remaining_self_loops(edge_index)
    source = edge_index[0]
    dest = edge_index[1]
    return scatter_('max', x[dest], source, dim_size=len(x))