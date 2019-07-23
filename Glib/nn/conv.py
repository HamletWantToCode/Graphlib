from ..base import BaseModel
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.inits import uniform

# import torchsnooper


class SGC_LL(MessagePassing, BaseModel):
    def __init__(self, in_channels, out_channels, K, alpha, root_weight=True, bias=True, **kwargs):
        super(SGC_LL, self).__init__(aggr='add', **kwargs)

        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        self.W_weight = Parameter(torch.Tensor(in_channels, in_channels))
        
        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)
        uniform(size, self.W_weight)
        uniform(size, self.root)

    @staticmethod
    def residue_norm(x, edge_index, num_nodes, W, alpha, dtype=None):
        row, col = edge_index

        # compute graph Laplacian
        edge_weight_L = torch.ones((edge_index.size(1), ),
                                  dtype=dtype,
                                  device=edge_index.device)

        edge_index_L, edge_weight_L = add_remaining_self_loops(
            edge_index, edge_weight_L, fill_value=0
        )
        row_L, col_L = edge_index_L

        deg = scatter_add(edge_weight_L, row_L, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        L = -deg_inv_sqrt[row_L] * edge_weight_L * deg_inv_sqrt[col_L]
        L[L==0] = 1

        # compute residue Laplacian
        tmp_row = torch.index_select(x, 0, row)
        tmp_col = torch.index_select(x, 0, col)
        diff = (tmp_row - tmp_col)
        diff_T = torch.transpose(diff, 1, 0)
        M = W.mm(torch.transpose(W, 1, 0))
        D2 = torch.einsum('ij,jk,ki->i', diff, M, diff_T)
        edge_weight_residue = torch.exp(-D2/2.0)

        _, edge_weight_residue = add_remaining_self_loops(
            edge_index, edge_weight_residue, fill_value=0
        )

        deg_Gauss = scatter_add(edge_weight_residue, row_L, dim=0, dim_size=num_nodes)
        deg_inv_sqrt_Gauss = deg.pow(-0.5)
        deg_inv_sqrt_Gauss[deg_inv_sqrt_Gauss == float('inf')] = 0

        L_residue = -deg_inv_sqrt_Gauss[row_L] * edge_weight_residue * deg_inv_sqrt_Gauss[col_L]
        L_residue[L_residue==0] = 1

        return edge_index_L, L + alpha*L_residue

    # @torchsnooper.snoop()
    def forward(self, x, edge_index):
        edge_index, norm = self.residue_norm(x, edge_index, x.size(0), self.W_weight, self.alpha, x.dtype)

        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.root is not None:
            out = out.mm(self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))