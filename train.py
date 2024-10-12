import sys
import os
import argparse
import random
import numpy as np
import torch 
from model.vae_cluster import Vae_Cluster_Es
from transformers import AutoTokenizer
from rich.console import Console
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader,Dataset
from utils.pepler_dataloader import Dataset_Rs_Pytorch,DataLoader_Rs
from collections import Counter
from transformers import AutoTokenizer,Trainer,TrainingArguments,DataCollatorWithPadding,EarlyStoppingCallback,DataCollatorForSeq2Seq
from datasets import load_from_disk
from utils.utils import TorchDataset2HuggingfaceDataset,plot_latent,RecTrainer,save_gate_index
from utils.prompt_process import Prompt_Process
from peft import LoraConfig, TaskType, get_peft_model
from model.moe_layer_llama import MoeBlock_RS
from model.vamoe import Vmoe_llama3

def train(model, train_dataset, eval_dataset, tokenizer, epoch, checkpoint_dir, args):
    trainer = RecTrainer(
        model             = model,
        train_dataset     = train_dataset,  
        eval_dataset      = eval_dataset,
        tokenizer         = tokenizer,
        data_collator     = DataCollatorForSeq2Seq(
            tokenizer     = tokenizer,
            padding       = True,
        ),
        save_lora         = True,
        args = TrainingArguments(

            output_dir                     = checkpoint_dir,
            save_strategy                  = 'steps',
            save_steps                     = 1000,
            per_device_train_batch_size    = 1,
            learning_rate                  = 3e-5,
            num_train_epochs               = 1,
            gradient_accumulation_steps    = 16,
            # --------- logging arguments --------- #
            logging_strategy               = 'steps',
            logging_steps                  = 10,
            report_to                      = 'tensorboard',
            save_safetensors               = True,

            max_grad_norm                  = 0.3,
            gradient_checkpointing         = True,
            # deepspeed                      = "",  
        )
    )

    print(len(trainer.train_dataset['input_ids'][0]),len(trainer.train_dataset['labels'][0]))
    print('start {} training!'.format(args.dataset))
    trainer.train()

    print('{} training done!'.format(args.dataset))

    # ====================== save model ===================== #
    # trainer.save_model(checkpoint_dir)
    print('{} model saved!'.format(args.dataset))

console = Console()
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VMoe_Rs')
    parser.add_argument('--dataset', type=str, default='TripAdvisor',
                        help='dataset name, ex: Amazon, Yelp, TripAdvisor')     
    parser.add_argument('--data_path', type=str, default='',
                        help='data path')            
    parser.add_argument('--index_dir', type=str, default='',
                        help='dataset index file')     
    parser.add_argument('--pretrain_epochs', type=int, default= 50,
                        help='epoch of pretrain GMM')     
    parser.add_argument('--latent_dim', type=int, default = 128,
                        help='latent dim')          
    parser.add_argument('--embedding_size', type=int, default = 768,
                        help='user-item embedding size')      
    parser.add_argument('--num_cluster', type=int, default = 5,
                        help='number of cluster')     
    parser.add_argument('--pretrain_model_path', type=str, default='',
                        help='local path of llm')   
    parser.add_argument('--batch_size', type=int, default = 4096,
                        help='batch size') 
    parser.add_argument('--cuda', action='store_true',default=True,
                        help='use CUDA')
    parser.add_argument('--pretrain_weight_save', type = str, default='',
                        help='path to save the pretraining model')
    parser.add_argument('--cluster_epoch', type=int, default = 30,
                        help='epoch of cluster')
    parser.add_argument('--lr', type=int, default =  0.00001,
                        help='Learning rate for training vae & gmm')
    parser.add_argument('--output_dir', type = str, default = '',
                        help='Explainable Model Training Results Storage Path')
    parser.add_argument('--llm_epoch', type = int, default = 3, help='epoch of llm')
    args = parser.parse_args()

    # ========================================================  Config Setting  ======================================================== 
    seed = 105
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
        device = 'cuda'
    else:
        device = 'cpu'

    if not os.path.exists(os.path.join(args.pretrain_weight_save, args.dataset)):
        os.mkdir(os.path.join(args.pretrain_weight_save,args.dataset))
        console.print(f'{args.dataset} Will be Save {os.path.join(args.pretrain_weight_save, args.dataset)}')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    console.print('Loading data...',style = 'bold green')
    max_text_length = 30

    corpus = DataLoader_Rs(args.data_path, args.index_dir, tokenizer, max_text_length)
    n_user = len(corpus.user_dict)
    n_item = len(corpus.item_dict)

    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        console.print('{:40} {}'.format(arg, getattr(args, arg)))
    console.print(f"user num: {n_user} item num: {n_item}")
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    
    # ========================================================  Pretraining       ======================================================== 
    vae_clu = Vae_Cluster_Es(n_user = n_user,n_item = n_item,args = args)
    with open(os.path.join(args.pretrain_weight_save, args.dataset, args.dataset + '_output.txt'), 'w') as f:
        f.write(str(vae_clu))
    vae_clu = vae_clu.to(device)
    vae_clu.pretrain(corpus = corpus, pretrain_epoch = args.pretrain_epochs)
    
    console.print(f'Pretraining finished....')
    # ========================================================  Cluster Training  ======================================================== 
    console.print(f'Cluster Training...')
    vae_clu.cluster_training(corpus = corpus, cluster_epoch = 100)


    console.print(f'Start Cluster Training......', style='bold red')
    cluster_epoch = args.cluster_epoch
    epoch_bar = tqdm(range(cluster_epoch))
    data_loader = DataLoader(Dataset_Rs_Pytorch(corpus.train),batch_size = args.batch_size, shuffle = True)
    losses = []                                                             
    # lr=0.001 better   lr is important,2e-3 lead to posterior collapse 🤡
    optimizer = torch.optim.Adam(vae_clu.parameters(),lr = args.lr)
    
    lr_s = StepLR(optimizer, step_size = 10, gamma = 0.5)
    print(f'len dataloader: {len(data_loader)}')

    scale_factor_kl = 0.01
    kl_increase = True
    for epoch in epoch_bar:
        # lr_s.step()
        epoch = epoch + 1
        loss_all = 0
        losses_epoch = 0.
        best_val_loss = float('inf')
        print(f'scale_factor_kl is {scale_factor_kl}')
        for batch_index,(user, item, rating, _, _) in enumerate(data_loader):
            user = user.to(device)
            item = item.to(device)
            rating = rating - 1
            rating = rating.to(device)
            # compute elbo loss -> batch loss
            loss = vae_clu.Elbo_loss(user, item, rating, scale_factor_kl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all += loss
        if epoch % 5 == 0: # scale_factor_kl 0.2 is better than 0.3
            # print('scale up scale_factor_kl')
            if kl_increase:
                scale_factor_kl += 0.005
                if scale_factor_kl >= 0.1:
                    scale_factor_kl = 0.1  
            plot_latent(vae_clu, data_loader, args, epoch)
            torch.save(vae_clu.state_dict(), os.path.join(args.pretrain_weight_save, args.dataset, args.dataset + '_' +f'cluster_{args.num_cluster}_epoch_{epoch}.pth'))
        
        losses_epoch = losses_epoch / len(data_loader) 

        if losses_epoch < best_val_loss:      
            best_val_loss = losses_epoch
            torch.save(vae_clu.state_dict(), os.path.join(args.pretrain_weight_save, args.dataset, args.dataset + '_' +f'cluster_{args.num_cluster}_best_weight.pth'))
            print(f'Saving Best Pretraining Model for loss {best_val_loss}')

        lr_s.step()
        print(f'Epoch {epoch} Loss: {loss_all.item() / len(data_loader)}')
        losses.append(loss_all.item() / len(data_loader))

        vae_clu.plot_loss_curve(losses, title=f'Cluster Training Loss Curve for {args.dataset}',save_path= os.path.join(args.pretrain_weight_save,args.dataset, args.dataset +'_'+ f'loss_cluster_{args.num_cluster}.png'))
        torch.save(vae_clu.state_dict(), os.path.join(args.pretrain_weight_save, args.dataset, args.dataset + '_' +f'cluster_{args.num_cluster}.pth'))
    console.print(f'Explaination Generate Training Start......',style = 'bold green')
    # ========================================================  Explaination Generate Training  ======================================================== 
    # construct Huggingface Dataset
    train_dataset = TorchDataset2HuggingfaceDataset(corpus.train, cache_dir = '' )
    eval_dataset  = TorchDataset2HuggingfaceDataset(corpus.valid, cache_dir = '' )
    test_dataset  = TorchDataset2HuggingfaceDataset(corpus.test, cache_dir = '' )

    # Mapping the dataset 
    # bound to set batched to False, data process is not batched ref: prompt_precess.py examples['rating'] >=3 positive

    print('Load the hf dataset...')
    train_dataset = train_dataset.map(
        Prompt_Process(tokenizer, 180),
        batched = False,
    )
    eval_dataset  = eval_dataset.map(
        Prompt_Process(tokenizer, 180),
        batched = False
    )
    test_dataset  = test_dataset.map(
        Prompt_Process(tokenizer, 180),
        batched = False
    )

    console.print(tokenizer.decode(train_dataset['input_ids'][0]),style='bold green')
    train_cluster_index = save_gate_index(train_dataset, vae_clu)
    print(len(train_dataset['input_ids'][0]),len(train_dataset['input_ids'][1]))
    lora_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM, 
        target_modules = ['q_proj','v_proj','k_proj','o_proj','user_embed','item_embed'],
        modules_to_save = ['f3','f1','f2','gate0','gate1','gate2','gate3','gate4','user_proj','item_proj'],
        inference_mode = False, 
        r = 8, 
        lora_alpha = 16, 
        lora_dropout = 0.1 
)
    from model.config_llama3 import llama_config
    
    config = llama_config
    print(config)

    user_embeds = vae_clu.encoder.user_embeddings
    item_embeds = vae_clu.encoder.item_embeddings

    user_embeds = user_embeds.to(torch.bfloat16)
    item_embeds = item_embeds.to(torch.bfloat16)

    vmoe_llama3 = Vmoe_llama3(config = config, tokenizer = tokenizer, gate_index_list = train_cluster_index, user_embed = user_embeds, item_embed = item_embeds, use_lora = False)

    model_llama3 = get_peft_model(vmoe_llama3,lora_config)
    vae_clu = vae_clu.to('cpu')
    del vae_clu
    torch.cuda.empty_cache()

    print('Already Freeze the user item embedding...')

    print(model_llama3.print_trainable_parameters())

    explain_checkpoint_dir = args.output_dir + '/explain'

    train(  
            epoch                   = args.llm_epoch, 
            model                   = model_llama3, 
            tokenizer               = tokenizer,
            train_dataset           = train_dataset,
            eval_dataset            = None,
            checkpoint_dir          = explain_checkpoint_dir,
            args                    = args
    )
    model_llama3.save_pretrained(explain_checkpoint_dir)
    print('Saved Model... && Training Done...')