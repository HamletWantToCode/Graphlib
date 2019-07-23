import torch
from ..nn import SGC_LL, graph_max_pool
from torch_geometric.nn import global_add_pool #, Set2Set
from torch.nn import BatchNorm1d
import torch.nn.functional as F

# import torchsnooper

class AGCN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 node_hidden_dim=15,
                 K=4,
                 alpha=0.5,
                 num_step_combo=2,
                #  num_step_set2set=6,
                 filter_in_channel=1,
                 filter_out_channel=1,
                 filer_kernel_size=3,
                 fcc1_hidden_dim=50,
                 fcc2_hidden_dim=15,
                 output_dim=12,
                 ):

        super(AGCN, self).__init__()

        self.num_step_combo = num_step_combo
        self.conv = SGC_LL(node_input_dim, node_hidden_dim, K, alpha)
        self.batch_norm = BatchNorm1d(node_hidden_dim)
        # self.set2set = Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
        self.filter = torch.nn.Conv1d(filter_in_channel, filter_out_channel, filer_kernel_size, padding=1)
        self.lin1 = torch.nn.Linear(node_hidden_dim, fcc1_hidden_dim)
        self.lin2 = torch.nn.Linear(fcc1_hidden_dim, fcc2_hidden_dim)
        self.output = torch.nn.Linear(fcc2_hidden_dim, output_dim)

    # @torchsnooper.snoop()
    def forward(self, data):
        out = data.x
        batch = data.batch
        edge_index = data.edge_index

        for i in range(self.num_step_combo):
            out = self.batch_norm(self.conv(out, edge_index))
            out = F.elu(out)
            out = graph_max_pool(out, edge_index)

        out = global_add_pool(out, batch)
        # out = self.set2set(out, batch)
        out = (self.filter(out.unsqueeze(dim=1))).squeeze(dim=1)
        
        out = F.relu(self.lin1(out))
        out = F.relu(self.lin2(out))
        out = self.output(out)
        return out
