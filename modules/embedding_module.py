import torch
from torch import nn
import numpy as np
import math
import pdb
from model.temporal_attention_SLADE import TemporalAttentionLayer_recovery


class EmbeddingModule(nn.Module):
    def __init__(self, memory, neighbor_finder, time_encoder, n_layers,
                n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                dropout):
        super(EmbeddingModule, self).__init__()
        self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20):
        pass
        

class GraphEmbedding(EmbeddingModule):
    def __init__(self, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
        super(GraphEmbedding, self).__init__(memory, neighbor_finder, time_encoder, n_layers,
                                            n_node_features, n_edge_features, n_time_features,
                                            embedding_dimension, device, dropout)
        self.use_memory = use_memory
        self.device = device
        self.memory_dropout = dropout
    
    def compute_recovery_memory_embedding(self, memory, source_nodes, timestamps, neighbors, neighbors_time, n_layers, n_neighbors=20):

        assert (n_layers >= 0)
        source_nodes_torch = source_nodes
        timestamps_torch = timestamps.unsqueeze(dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
        timestamps_torch))


        neighbors_torch = neighbors
        edge_deltas_torch = timestamps.unsqueeze(dim=1) - neighbors_time
        neighbors = neighbors.flatten()

        mem_neighbor_embeddings = memory[neighbors,:]
        effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        mem_neighbor_embeddings = mem_neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)

        edge_time_embeddings = self.time_encoder(edge_deltas_torch)
        mask = neighbors_torch == 0

        if self.training:
            memory_mask = (torch.rand(mask.size()) < self.memory_dropout).to(self.device)
            mask = torch.logical_or(mask, memory_mask) 

        mem_recovery_embedding = self.aggregate(n_layers, source_nodes_time_embedding,
                                      mem_neighbor_embeddings,
                                      edge_time_embeddings,
                                      mask)
        return mem_recovery_embedding
    
    def aggregate(self, n_layers, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, mask):
        return None
    

class GraphAttentionEmbedding_recovery(GraphEmbedding):
    def __init__(self, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
        super(GraphAttentionEmbedding_recovery, self).__init__(memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory)
        
        self.attention_models = TemporalAttentionLayer_recovery(
                                                        n_node_features=n_node_features,
                                                        n_neighbors_features=n_node_features,
                                                        time_dim=n_time_features,
                                                        n_head=n_heads,
                                                        dropout=dropout,
                                                        output_dimension=n_node_features)

    def aggregate(self, n_layer, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, mask):
        
        attention_model = self.attention_models
        source_embedding, _ = attention_model(source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          mask)
        return source_embedding

def get_embedding_module(module_type, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):  
    if module_type == "graph_attention_recovery":
        return GraphAttentionEmbedding_recovery(memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))