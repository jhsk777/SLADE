import logging
import numpy as np
import torch

from utils.utils import cosine_similarity
from modules.memory import Memory
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode, Fixed_time_encode
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
import pdb

class SLADE_TGN(torch.nn.Module):
    def __init__(self, neighbor_finder, device, n_nodes, n_edges, n_layers=1, 
                 n_heads=2, dropout=0.1, message_dimension=128, memory_dimension=256,
                 n_neighbors=10, memory_agg_type='TGAT',negative_memory_type='train', message_updater='mlp', memory_updater='gru',
                 src_reg_factor=1, dst_reg_factor=1, only_drift_loss=False, only_recovery_loss=False):
        super(SLADE_TGN, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)


        self.n_node_features = memory_dimension
        self.n_nodes = n_nodes + 1 # first node memory is empty (because of zero neighbor masks)
        self.n_edge_features = memory_dimension
        self.n_edges = n_edges
        
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.src_reg_factor = src_reg_factor
        self.dst_reg_factor = dst_reg_factor
        self.only_drift_loss = only_drift_loss
        self.only_recovery_loss = only_recovery_loss
        
        self.time_encoder = Fixed_time_encode(dimension=self.n_node_features)
        
        self.memory = None
        self.bias_src_memory = nn.Parameter(torch.zeros(memory_dimension), requires_grad=False)
        self.bias_dst_memory = nn.Parameter(torch.zeros(memory_dimension), requires_grad=False)

        self.memory_agg_type = memory_agg_type
        self.negative_memory_type = negative_memory_type

        self.memory_dimension = memory_dimension
        self.raw_message_dimension = self.memory_dimension + self.time_encoder.dimension

        if message_updater=='mlp':
            self.message_dimension = message_dimension
        else:
            self.message_dimension = self.raw_message_dimension

        self.memory = Memory(n_nodes=self.n_nodes,
                            memory_dimension=self.memory_dimension,
                            input_dimension=self.message_dimension,
                            message_dimension=self.message_dimension,
                            raw_message_dimension=self.raw_message_dimension,
                            device=device)
        
        
        self.message_function = get_message_function(module_type=message_updater,
                                                    raw_message_dimension=self.raw_message_dimension,
                                                    message_dimension=self.message_dimension)
        self.memory_updater = get_memory_updater(module_type=memory_updater,
                                                memory=self.memory,
                                                message_dimension=self.message_dimension,
                                                memory_dimension=self.memory_dimension,
                                                device=device)
        
        self.embedding_module_recovery = get_embedding_module(module_type="graph_attention_recovery",
                                                    memory=self.memory,
                                                    neighbor_finder=self.neighbor_finder,
                                                    time_encoder=self.time_encoder,
                                                    n_layers=self.n_layers,
                                                    n_node_features=self.n_node_features,
                                                    n_edge_features=self.n_edge_features,
                                                    n_time_features=self.n_node_features,
                                                    embedding_dimension=self.embedding_dimension,
                                                    device=self.device,
                                                    n_heads=n_heads, dropout=dropout,
                                                    use_memory=True,
                                                    n_neighbors=self.n_neighbors)
        



    def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times,
                                    src_neighbors, dst_neighbors,src_neighbors_time, dst_neighbors_time, n_neighbors, test=False):
        
        n_samples = len(source_nodes)
        nodes = torch.concat([source_nodes, destination_nodes])
        timestamps = torch.concat([edge_times, edge_times])
        neighbors = torch.concat([src_neighbors, dst_neighbors])
        neighbors_time = torch.concat([src_neighbors_time, dst_neighbors_time])

        memory = None
        if test:
            all_nodes = torch.concat([nodes,neighbors.reshape(-1)])
            memory, _ = self.get_updated_memory_tensor(all_nodes, self.memory.messages_tensor, self.memory.messages_time)
        else:
            memory, _ = self.get_updated_memory_tensor(torch.arange(1, self.n_nodes), self.memory.messages_tensor, self.memory.messages_time)
        

        pos_node_embedding, dst_node_embedding = None, None
        if self.memory_agg_type == 'TGAT':
            node_embedding_TGAT = self.embedding_module_recovery.compute_recovery_memory_embedding(memory=memory,
                                                                    source_nodes=nodes,
                                                                    timestamps=timestamps, neighbors=neighbors, neighbors_time=neighbors_time,
                                                                    n_layers=self.n_layers,
                                                                    n_neighbors=n_neighbors)
            pos_node_embedding = node_embedding_TGAT[:n_samples]
            dst_node_embedding = node_embedding_TGAT[n_samples:]
        else:
            raise AssertionError('memory agg type is wrong!')
        
        node_memory = memory[source_nodes]
        dst_node_memory = memory[destination_nodes]

        self.update_memory_tensor(nodes, self.memory.messages_tensor, self.memory.messages_time)
        self.memory.clear_messages(nodes)

        unique_sources, source_message, source_message_ts = self.get_raw_messages(source_nodes, destination_nodes, edge_times)        
        unique_destinations, dst_message, dst_message_ts = self.get_raw_messages(destination_nodes, source_nodes, edge_times)
        
        self.memory.store_raw_messages(unique_sources, source_message, source_message_ts)
        self.memory.store_raw_messages(unique_destinations, dst_message, dst_message_ts)
        return node_memory, pos_node_embedding, dst_node_memory, dst_node_embedding
    

    def compute_node_diff_score(self, source_nodes, destination_nodes, edge_times, src_neighbors, dst_neighbors, src_neighbors_time, dst_neighbors_time, n_neighbors, seen_nodes):
        n_samples = len(source_nodes)
        prev_memory = self.memory.get_memory(source_nodes)
        prev_dst_memory = self.memory.get_memory(destination_nodes)
            
        
        node_memory, pos_node_embedding, dst_node_memory, dst_node_embedding = self.compute_temporal_embeddings(source_nodes, destination_nodes, edge_times, src_neighbors, dst_neighbors, src_neighbors_time, dst_neighbors_time, n_neighbors)
        if self.negative_memory_type == 'random':
            random_node = np.random.randint(0, self.n_nodes, size=n_samples)
            negative_memory = self.memory.memory[random_node]
        elif self.negative_memory_type == 'train':
            negative_memory = self.memory.memory[seen_nodes]
        else:
            raise AssertionError("negative memory type is wrong")
        

        positive_recovery_score = torch.exp(torch.diag(cosine_similarity(pos_node_embedding, node_memory)).reshape(-1))
        negative_recovery_score = torch.exp(cosine_similarity(pos_node_embedding, negative_memory)).sum(dim=1)

        positive_drift_score = (torch.exp(torch.diag(cosine_similarity(node_memory, prev_memory)).reshape(-1)))
        negative_drift_score = torch.exp(cosine_similarity(node_memory, negative_memory)).sum(dim=1)

        dst_positive_recovery_score = torch.exp(torch.diag(cosine_similarity(dst_node_embedding, dst_node_memory)).reshape(-1)) 
        dst_negative_recovery_score = torch.exp(cosine_similarity(dst_node_embedding, negative_memory)).sum(dim=1)

        dst_positive_drift_score = (torch.exp(torch.diag(cosine_similarity(dst_node_memory, prev_dst_memory)).reshape(-1)))
        dst_negative_drift_score = torch.exp(cosine_similarity(dst_node_memory, negative_memory)).sum(dim=1)

        
        if self.only_drift_loss:      # only memory drift loss  
            contrastive_loss = - torch.log(positive_drift_score/(negative_drift_score)) \
                            - torch.log(dst_positive_drift_score/(dst_negative_drift_score))
            
        elif self.only_recovery_loss: # only memory reconstruciton loss
            contrastive_loss = - torch.log(positive_recovery_score/(negative_recovery_score))\
                            - torch.log(dst_positive_recovery_score/(dst_negative_recovery_score))
        else:                         # original
            contrastive_loss = - torch.log(positive_recovery_score/(negative_recovery_score)) * self.src_reg_factor  \
                                - torch.log(dst_positive_recovery_score/(dst_negative_recovery_score)) * self.dst_reg_factor \
                                - torch.log(positive_drift_score/(negative_drift_score)) \
                                - torch.log(dst_positive_drift_score/(dst_negative_drift_score))


        contrastive_loss = contrastive_loss.mean()

        return positive_recovery_score, positive_drift_score, dst_positive_recovery_score, dst_positive_drift_score, contrastive_loss
    


    def compute_anomaly_score(self, source_nodes, destination_nodes, edge_times, src_neighbors, dst_neighbors, src_neighbors_time, dst_neighbors_time, n_neighbors):
        prev_memory = self.memory.get_memory(source_nodes)
        prev_dst_memory = self.memory.get_memory(destination_nodes)
        node_memory, pos_node_embedding, dst_node_memory, dst_node_embedding = self.compute_temporal_embeddings(source_nodes, destination_nodes, edge_times, src_neighbors, dst_neighbors, 
                                                                                                                src_neighbors_time, dst_neighbors_time, n_neighbors, test=True)      
        

        positive_recovery_score = torch.diag(cosine_similarity(pos_node_embedding, node_memory)).reshape(-1)
        dst_positive_recovery_score = torch.diag(cosine_similarity(dst_node_embedding, dst_node_memory)).reshape(-1)
        positive_drift_score = torch.diag(cosine_similarity(node_memory, prev_memory)).reshape(-1)
        dst_positive_drift_score = torch.diag(cosine_similarity(dst_node_memory, prev_dst_memory)).reshape(-1)

        return positive_recovery_score, positive_drift_score, dst_positive_recovery_score, dst_positive_drift_score                          

    

 
    def get_raw_messages(self, source_nodes, destination_nodes, edge_times):
        destination_memory = self.memory.get_memory(destination_nodes)

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
        source_nodes), -1)
        source_message = torch.cat([destination_memory, source_time_delta_encoding], dim=1)
        source_nodes_torch = source_nodes
        (nid, idx) = torch.unique(source_nodes_torch, return_inverse=True)
        message = scatter(source_message, idx, reduce='mean', dim=0)
        message_ts = scatter(edge_times, idx, reduce='max')
        unique_sources = nid

        return unique_sources, message, message_ts


    def update_memory_tensor(self, nodes, messages_tensor, messages_ts_tensor):
        # Aggregate messages for the same nodes

        unique_node_ids = torch.unique(nodes).to(self.device)
        mask = (messages_ts_tensor[unique_node_ids] != 0)

        masked_unique_nodes = unique_node_ids[mask]
        unique_node_messages = messages_tensor[unique_node_ids][mask]
        unique_node_ts = messages_ts_tensor[unique_node_ids][mask]
        
        if len(masked_unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_node_messages)
        else:
            unique_messages = None

            # Update the memory with the aggregated messages
        self.memory_updater.update_memory(masked_unique_nodes, unique_messages,
                                            timestamps=unique_node_ts)
            
    def get_updated_memory_tensor(self, nodes, messages_tensor, messages_ts_tensor):
        # Aggregate messages for the same nodes

        unique_node_ids = torch.unique(nodes).to(self.device)
        mask = (messages_ts_tensor[unique_node_ids] != 0)

        masked_unique_nodes = unique_node_ids[mask]
        unique_node_messages = messages_tensor[unique_node_ids][mask]
        unique_node_ts = messages_ts_tensor[unique_node_ids][mask]

        if len(masked_unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_node_messages)
        else:
            unique_messages = None
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(masked_unique_nodes,
                                                                                    unique_messages,
                                                                                    timestamps=unique_node_ts)
        return updated_memory, updated_last_update

    
    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module_recovery.neighbor_finder = neighbor_finder





