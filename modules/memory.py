import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy
import torch.nn.init as init
import numpy as np


class Memory(nn.Module):
    def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None, raw_message_dimension=None,
               device="cpu", combination_method='sum',use_random_memory=False):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.raw_message_dimension = raw_message_dimension
        self.device = device
        self.use_random_memory = use_random_memory

        self.combination_method = combination_method

        self.__init_memory__()
    
    def __init_memory__(self):
        if self.use_random_memory:
            self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device), requires_grad=False) 
            init.xavier_normal_(self.memory)
        else:
            self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False) 
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False)

        self.messages_tensor = nn.Parameter(torch.zeros(self.n_nodes, self.raw_message_dimension).to(self.device),requires_grad=False)
        self.messages_time = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),requires_grad=False)

    def store_raw_messages(self, nodes, messages, message_ts):
        self.messages_tensor[nodes] = messages
        self.messages_time[nodes] = message_ts

    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]
    
    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def detach_memory(self):
        self.memory.detach_()
        self.messages_tensor.detach_()
        self.messages_time.detach_()



    def clear_messages(self, nodes):

        unique_nodes = torch.unique(nodes)
        self.messages_tensor[unique_nodes].zero_()
        self.messages_time[unique_nodes].zero_()