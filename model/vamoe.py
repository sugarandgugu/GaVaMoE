import os
import sys
import torch.nn as nn
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from vae_cluster import Vae_Cluster_Es
import warnings
warnings.filterwarnings('ignore')

from transformers import LlamaPreTrainedModel,AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from moe_layer_llama import MoeBlock_RS

class Vmoe_llama3(LlamaPreTrainedModel):
    def __init__(self, config, tokenizer, gate_index_list, user_embed, item_embed, use_lora = False):
        super().__init__(config)
        
        self.config = config
        self.gate_index_list = gate_index_list
        self.tokenizer = tokenizer
        self.use_lora = use_lora

        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.bfloat16,
                                                    config = self.config)
        self.change_mlp4moe()
        print(self.model)
        # useless
        if self.use_lora:
            pass
            
        if user_embed is not None and item_embed is not None:
            self.user_embed = user_embed
            self.item_embed = item_embed
            with torch.no_grad():  
                self.user_embed.weight.data.copy_(user_embed.weight.data)
                self.item_embed.weight.data.copy_(item_embed.weight.data)
        else:
            # replace different dataset 
            self.user_embed = nn.Embedding(9765, 768).to(torch.bfloat16)
            self.item_embed = nn.Embedding(6280, 768).to(torch.bfloat16)


        self.user_proj = nn.Linear(768, config.hidden_size).to(torch.bfloat16)
        self.item_proj = nn.Linear(768, config.hidden_size).to(torch.bfloat16)
        
        self.index_count = 0

        self.post_init()

    def change_mlp4moe(self):
        # for the modulelist of llama
        count = 0 
        for block in self.model.model.layers:
            block.mlp = MoeBlock_RS(config = self.config, cluster_index_list = self.gate_index_list, dataset_num = 100).to(torch.bfloat16)
            print(f'already replace mlp to moe {count}')
            count = count + 1
        return
        # model.base_model.model.model.model
    '''
        BATCH_SIZE must be 1, or forward func is wrong.
    '''
    def forward(
        self, 
        user, 
        item, 
        input_ids, 
        attention_mask,
        labels,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None
        ):

        for block in self.model.model.layers:
            block.mlp.cluster_index_count  = block.mlp.cluster_index_count + 1
        
        device = self.model.device
        batch_size = input_ids.size(0)

        user = torch.tensor(user).contiguous().to(device)
        item = torch.tensor(item).contiguous().to(device)
        
        user_embed = self.user_embed(user).to(device)
        item_embed = self.item_embed(item).to(device)
        
        self.user_proj = self.user_proj.to(device)
        self.item_proj = self.item_proj.to(device)

        user_embeds = self.user_proj(user_embed)
        item_embeds = self.item_proj(item_embed)
        
        # # ITEM: 37032 USER: 14194
        user_index = torch.where(input_ids[0] == 14194)
        item_index = torch.where(input_ids[0] == 37032)

        assert user_index is not None and item_index is not None, "Indices must not be None"
        assert len(user_index[0]) < 2 and len(item_index[0]) < 2, "Indices must be less than 2"

        user_index = user_index[0][0].item()
        item_index = item_index[0][0].item()

        if self.use_lora:
            prompt_embeds = self.model.base_model.model.model.embed_tokens(input_ids)
        else:
            prompt_embeds = self.model.model.embed_tokens(input_ids)

        prompt_embeds[0][item_index] = item_embeds
        prompt_embeds[0][user_index] = user_embeds

        final_embed = prompt_embeds

        llm_output = self.model(inputs_embeds = final_embed, attention_mask = attention_mask, labels = labels)
        loss = llm_output.loss

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=llm_output.logits,
            past_key_values=llm_output.past_key_values,
            hidden_states=llm_output.hidden_states,
            attentions=llm_output.attentions,
        )

    def generate(self, user, item, input_ids):
        device = input_ids.device
        batch_size = input_ids.size(0)
        for block in self.model.model.layers:
            block.mlp.cluster_index_count  = block.mlp.cluster_index_count + 1
        user = torch.tensor(user).contiguous().to(device)
        item = torch.tensor(item).contiguous().to(device)
        
        user_embed = self.user_embed(user)
        item_embed = self.item_embed(item)
        
        user_embeds = self.user_proj(user_embed)
        item_embeds = self.item_proj(item_embed)

        user_index = torch.where(input_ids[0] == 14194)
        item_index = torch.where(input_ids[0] == 37032)

        user_index = user_index[0][0].item()
        item_index = item_index[0][0].item()

        # print(self.model)
        prompt_embeds = self.model.model.embed_tokens(input_ids)

        prompt_embeds[0][item_index] = item_embeds
        prompt_embeds[0][user_index] = user_embeds

        final_embed = prompt_embeds
        
        output = self.model.generate(
                inputs_embeds   = final_embed,
                do_sample       = True, 
                # tokenizer.get_vocab()["<|eot_id|>"] -> 128001
                pad_token_id    = 128001,
                eos_token_id    = 128001,
                max_new_tokens  = 30,
                temperature     = 0.7,
        )
        return output
