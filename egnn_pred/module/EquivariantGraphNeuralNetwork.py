import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch


class EGCL(MessagePassing):
    def __init__(
        self,
        m_input,
        m_hidden,
        m_output,
        x_input,
        x_hidden,
        x_output,
        h_input,
        h_hidden,
        h_output,
        flow="target_to_source",
        aggr="mean",
        activation="SiLU",
    ):
        super(EGCL, self).__init__(aggr=aggr, flow=flow)

        self.mlp_m = nn.Sequential(
            nn.Linear(m_input, m_hidden),
            nn.SiLU(),
            nn.Linear(m_hidden, m_output),
            nn.SiLU(),
        )
        self.mlp_h = nn.Sequential(
            nn.Linear(h_input, h_hidden),
            nn.SiLU(),
            nn.Linear(h_hidden, h_output),
        )
        self.attention = nn.Sequential(
            nn.Linear(m_output, 1),
            nn.Sigmoid(),
        )

    def message(self, h_i, h_j, attention):
        message_input = torch.cat((h_i, h_j), dim=1)
        out = self.mlp_m(message_input)
        if attention:
            out = out * self.attention(out)
        return out

    def forward(self, edge_index, h, coords=None):
        sum_message = self.propagate(
            edge_index=edge_index,
            h=h,
            attention=True,
        )
        updated_h = self.mlp_h(torch.cat((h, sum_message), dim=1))
        return updated_h


class EquivariantGNN(nn.Module):
    def __init__(
        self,
        L,
        m_input,
        m_hidden,
        m_output,
        x_input,
        x_hidden,
        x_output,
        h_input,
        h_hidden,
        h_output,
        total_output_dim,
    ):
        super(EquivariantGNN, self).__init__()
        self.L = L
        self.egcl_list = nn.ModuleList(
            [
                EGCL(
                    m_input,
                    m_hidden,
                    m_output,
                    x_input,
                    x_hidden,
                    x_output,
                    h_input,
                    h_hidden,
                    h_output,
                )
                for _ in range(L)
            ]
        )
        self.h_embedding_out = nn.Sequential(
            nn.Linear(h_output, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, total_output_dim),
        )

    def forward(self, edge_index, h, x=None):
        for layer in self.egcl_list:
            h = layer(edge_index, h)

        output_h = self.h_embedding_out(h)
        if x is None:
            return output_h
        return output_h, x
