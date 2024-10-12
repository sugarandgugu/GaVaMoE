'''
    utlls.py: tool class
'''
import os
import re
import torch
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from transformers import PreTrainedTokenizerBase
from datasets import Dataset as HFDataset

from transformers.trainer_utils import EvalLoopOutput
from transformers import Trainer
from transformers.utils import logging
from torch.utils.data import SequentialSampler

'''
    process_explain_data_fun: 
        args:
            examples: single data
        pad、tokenize、add bos eos token
'''
def TorchDataset2HuggingfaceDataset(torch_dataset, cache_dir = None):
    generator = lambda: (sample for sample in torch_dataset)   
    return HFDataset.from_generator(generator, cache_dir=cache_dir)

def process_fun(examples):
    # examples['text'] = '<>'
    encode_inputs = tokenizer(examples['text'], max_length = 20, truncation = True)
    encode_inputs["user"] = examples["user"]
    encode_inputs["item"] = examples["item"]
    # encode_inputs["rating"] = examples["rating"]
    for key, value in tokenizer(tokenizer.bos_token).items():
        encode_inputs[key] = value + encode_inputs[key]

    for key, value in tokenizer(tokenizer.eos_token).items():
        encode_inputs[key] = encode_inputs[key] + value
        
    return encode_inputs

class Process_Explain_data:
    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase], max_seq_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, examples):
        model_inputs = self.tokenizer(examples["explanation"], 
                                      max_length=self.max_seq_length,
                                      truncation=True)
        model_inputs["user"] = examples["user"]
        model_inputs["item"] = examples["item"]
        model_inputs["rating"] = examples["rating"]
        
        # add prefix and postfix key: input_ids 
        for key, value in self.tokenizer(self.tokenizer.bos_token).items():
            model_inputs[key] = value + model_inputs[key]

        for key, value in self.tokenizer(self.tokenizer.eos_token).items():
            model_inputs[key] = model_inputs[key] + value

        # until this step, the length of each example input_ids is not equal
        return model_inputs



def plot_latent(vae_clu, data_loader, args, epoch):
    with torch.no_grad():
        Z = []
        Y = []
        vae_clu.eval()
        
        for batch_index,(user, item, rating, _, _) in enumerate(data_loader):
            user = user.to('cuda')
            item = item.to('cuda')
            z1, z2 = vae_clu.encoder(user, item)
            y = vae_clu.predict_cluster_index(user,item)
            Y.append(torch.tensor(y))
            Z.append(z1)
        # [batch, latent_dim]
        Z = torch.cat(Z, 0).detach().cpu().numpy()
        Y = torch.cat(Y, 0).detach().cpu().numpy()
        index_counts = Counter(Y)
        for index, count in index_counts.items():
            print(f"Cluster {index} appears {count} times.")

        print(f'🤡🤡🤡 Ploting Latent Space for {args.dataset}')
        num_samples_per_cluster = 300
        indices = []
        for i in range(args.num_cluster):
            indices_in_cluster = np.where(Y == i)[0]
            selected_indices = np.random.choice(indices_in_cluster, num_samples_per_cluster, replace=True)
            indices.extend(selected_indices)

        selected_Z = Z[indices]

        tsne = TSNE(n_components=2, random_state=42)  
        Z_2d = tsne.fit_transform(selected_Z)
        selected_pre = Y[indices]


        plt.figure(figsize=(10, 8))
        plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c = selected_pre, cmap='viridis', alpha=0.6)

        plt.colorbar()  
        plt.title(f't-SNE Visualization of Latent Space for {args.dataset}') 
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.savefig(os.path.join(args.pretrain_weight_save,args.dataset, args.dataset + '_' + f'latent_vis_cluster_{args.num_cluster}_epoch_{epoch}.png'),dpi=300)
        print(f'Plot Latent Space Done for {epoch}')



def dict_extend(dict, key, value):
    """
        extend the list value of key in dict
    """
    if key in dict and isinstance(value,list):
            dict[key].extend(value)
    else:
        dict[key] = value 

def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    Credit: AllenNLP
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


class RecTrainer(Trainer):
    def __init__(self, *args, save_lora = True, **kwargs):
        self.save_lora = save_lora
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        
        return SequentialSampler(self.train_dataset)


import torch
from torch.utils.data import DataLoader


def save_gate_index(hf_dataset, vae_clu, batch_size=1000):
    cluster_index_list = []

    hf_dataset = hf_dataset.remove_columns(['labels','feature', 'input_ids', 'attention_mask','rating'])
    data_loader = DataLoader(hf_dataset, batch_size=batch_size, shuffle=False)

    print('Processing the gate index...')
    total_batches = len(data_loader)
    processed_batches = 0

    for batch in data_loader:
        users = torch.tensor(batch['user']).to(vae_clu.device)
        items = torch.tensor(batch['item']).to(vae_clu.device)

        indices = vae_clu.predict_cluster_index(users, items)
        cluster_index_list.extend(indices.tolist()) 
    
        processed_batches += 1
        if processed_batches % 1000 == 0:
            print(f'process {processed_batches} / {total_batches}')
    
    print(f'Save Gate Index List Length: {len(cluster_index_list)}')
    return cluster_index_list


def postprocessing(string):
    '''
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub('\(', ' ( ', string)
    string = re.sub('\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string