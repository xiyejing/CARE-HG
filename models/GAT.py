import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import dgl

from model.PositionWiseGATConv import PositionWiseGATConv
from utils.global_variables import GRAPH_CONFIG_T, HYPERGRAPH_CONFIG_T, HYPERGRAPH_CONFIG_T_A, HYPERGRAPH_CONFIG_T_A_V

class GATLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout=0.2):
        super(GATLayer, self).__init__()

        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(PositionWiseGATConv(in_feats = in_size, out_feats = out_size, num_heads = layer_num_heads, 
                                           feat_drop = dropout, attn_drop = dropout, activation =F.leaky_relu,
                                           allow_zero_in_degree=True))
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self._cached_graph = None 
        self._cached_coalesced_graph = {} 

    def forward(self, g, h_dict): 
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, [meta_path])
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]

            edge_weight = None
            if 'edge_weight' in new_g.edata:
                edge_weight = new_g.edata['edge_weight']

            if meta_path[0] == meta_path[-1]:  
                hidden_embeddings = self.gat_layers[i](new_g, h_dict[meta_path[0]], edge_weight).flatten(1)
            else:
                hidden_embeddings = self.gat_layers[i](new_g, (h_dict[meta_path[0]], h_dict[meta_path[-1]]), edge_weight).flatten(1)

            new_h_dict = {}
            for ntype in h_dict.keys():
                if ntype == meta_path[-1]:
                    new_h_dict[ntype] = hidden_embeddings  
                else:
                    new_h_dict[ntype] = h_dict[ntype]  

            concat_all = torch.cat(
                [new_h_dict[ntype] for ntype in HYPERGRAPH_CONFIG_T['node_types']],
                dim=0
            )
            semantic_embeddings.append(concat_all)
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1) # Node_num * Meta_path_num * Embedding_dim
        return self.semantic_attention(semantic_embeddings)
    



class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0) 
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)