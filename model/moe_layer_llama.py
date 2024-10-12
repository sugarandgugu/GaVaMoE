import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Optional, Tuple
import torch.nn.functional as F
from transformers import LlamaConfig

class SparseTop2MLP(nn.Module):
    def __init__(self, config, intermediate_size = None):
        super().__init__()
        
        self.ffn_dim = intermediate_size
        self.hidden_dim = config.hidden_size

        self.f1 = nn.Linear(self.hidden_dim, self.ffn_dim,    bias = False)
        self.f2 = nn.Linear(self.ffn_dim,    self.hidden_dim, bias = False)
        self.f3 = nn.Linear(self.hidden_dim, self.ffn_dim,    bias = False)

        self.act = nn.SiLU()

    def forward(self, hidden_state):

        x = self.act(self.f1(hidden_state) * self.f3(hidden_state))
        x = self.f2(x)
        return x

class MoeBlock_RS(nn.Module):

    def __init__(self, config, cluster_index_list, dataset_num):
        super().__init__()
        self.ffn_dim = 1280  # 14336
        self.hidden_dim = config.hidden_size
        self.num_experts = 12
        self.top_k = 2
        self.num_cluster = 5

        self.gate = nn.ModuleDict({f"gate{i}": nn.Linear(self.hidden_dim, self.num_experts, bias = False) for i in range(self.num_cluster)})

        self.experts = nn.ModuleList([SparseTop2MLP(config) for _ in range(self.num_experts)])
        self.cluster_index_list = cluster_index_list 
        self.foward_count = 0   
        self.cluster_index_count = 0    
        self.dataset_num = dataset_num         
    '''
        input: 
            hidden_state: Transformer FFN
            cluster_index: index for choose which gate to use
                ex: cluster 0 : gate 0
                    cluster 1 : gate 1
                    ...
                    cluster n : gate n
                shape: [batch_size]
    '''
    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        router_logits_list = []
        for i,idx in enumerate(range(batch_size)):
            gate_index = self.cluster_index_list[self.cluster_index_count-1]
            routing_logits = self.gate['gate{}'.format(gate_index)](hidden_states[i])
            router_logits_list.append(routing_logits)

        router_logits = torch.stack(router_logits_list).view(-1, self.num_experts)
        hidden_states = hidden_states.view(-1, hidden_dim)
        routing_weights = F.softmax(router_logits, dim = -1)
        # select top2 experts
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim = -1) 
        # fusing weight && add
        routing_weights = routing_weights / torch.sum(routing_weights, dim = -1, keepdim = True).to(hidden_states.dtype)
        #  init maxtrix to save result
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),dtype=hidden_states.dtype,device=hidden_states.device
        )
        # for efficiency, calculate the result one time using the mask
        expert_mask = nn.functional.one_hot(selected_experts, num_classes = self.num_experts)
        # [20,2,8] ---> [8,2,20]
        expert_mask = expert_mask.permute(2, 1, 0)
        for expert_index in range(self.num_experts):
            expert_layer = self.experts[expert_index]
            idx, top_x = torch.where(expert_mask[expert_index])
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = hidden_states[None,top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list,idx_list, None]
            current_hidden_states = current_hidden_states.to(hidden_states.dtype)

            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        final_hidden_states = final_hidden_states.reshape(batch_size,seq_len,hidden_dim)

        return final_hidden_states


if __name__ == '__main__':
    import random 
    config = LlamaConfig()
    cluster_index_list = [random.randint(0, 4) for _ in range(10)]
    print(f'index{cluster_index_list}')
    moe_block = MoeBlock_RS(config, cluster_index_list)
    test_tensor = torch.randn(5,10,4096)
    out = moe_block(test_tensor)
    # print(out)
    print(out.shape)
