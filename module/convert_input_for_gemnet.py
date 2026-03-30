import os, torch
import numpy as np
import sys

class GraphToInputForGemnet():
    def __init__(self):
        self.graph = None
        self.input = None
    
    def set_graph(self, graph):
        self.graph = graph
        self.input = {
            "Z": None,
            "R": None,
            "spectrum": None,
            "time": None,
            "id_a": None,
            "id_c": None,
            "id_undir": None,
            "id_swap": None,
            "id3_expand_ba": None,
            "id3_reduce ca": None,
            "batch_seg": None,
            "Kidx3": None,
        }
        
    
    def assign_input(self):
        if self.graph is None:
            raise ValueError("Graph is not set. Please set the graph before getting input.")
        if not self.graph.pos_at_t.requires_grad:
            self.graph.pos_at_t.requires_grad = True        
        self.input["R"] = self.graph.pos_at_t
        """
        Z_list = []
        for i in range(self.graph.x.shape[0]):
            if torch.equal(self.graph.x[i].cpu(), torch.tensor([1, 0],dtype=torch.long).cpu()):
                Z_list.append(8)
            elif torch.equal(self.graph.x [i].cpu(), torch.tensor([0, 1], dtype=torch.long).cpu()):
                Z_list.append(14)
        self.input["Z"] = torch.tensor(Z_list, dtype=torch.long)
        """
        self.input["Z"] = self.graph.h_at_t
        self.input["spectrum"] = self.graph.spectrum
        self.input["time"] = self.graph.each_time_list / 1000

        edges = self.graph.edge_index
        mask = edges[0] < edges[1]
        unduplicated_edges = edges[:, mask]
        reversed_edges = torch.stack([unduplicated_edges[1], unduplicated_edges[0]], dim=0)
        edge_index = torch.cat([unduplicated_edges, reversed_edges], dim=1)
        self.input["id_a"] = edge_index[0]
        self.input["id_c"] = edge_index[1]
        indices = torch.tensor([i for i in range(len(reversed_edges[0]))], dtype=torch.long)
        self.input["id_undir"] = torch.cat([indices, indices], dim=0)

        self.input["id_swap"] = torch.cat([indices + len(reversed_edges[0]), indices], dim=0) 

        expand_ba, reduce_ca = [], []
        id_t = edge_index[0]
        id_s = edge_index[1]
        for i in range(id_t.max()+1):
            edge_indices = torch.where(id_t == i)[0]
            for j in range(edge_indices.shape[0]):
                for k in range(edge_indices.shape[0]):
                    if j != k:
                        expand_ba.append(edge_indices[j].item())
                        reduce_ca.append(edge_indices[k].item())
        idx_sorted = torch.argsort(torch.tensor(reduce_ca, dtype=torch.long))
        id3_expand_ba = torch.tensor(expand_ba, dtype=torch.long)[idx_sorted]
        id3_reduce_ca = torch.tensor(reduce_ca, dtype=torch.long)[idx_sorted]
        _, K = torch.unique(id3_reduce_ca, return_counts=True)

        self.input["id3_expand_ba"] = id3_expand_ba
        self.input["id3_reduce_ca"] = id3_reduce_ca

        kidx3 = []
        for i in range(K.shape[0]):
            kidx3 = kidx3 + list(range(K[i].item()))
        self.input["Kidx3"] = torch.tensor(kidx3, dtype=torch.long)

        self.input["batch_seg"] = torch.tensor(self.graph.batch, dtype=torch.long)

        return self.input

    def convert_input_for_gemnet(self, graph):
        self.set_graph(graph)
        return self.assign_input()
    