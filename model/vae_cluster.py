'''
    vae_cluster.py: 
        1、 Pretraining VAE to get prior
        2、design Encoder and Decoder for rating construct
        3、design elbo loss
'''
import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import sys
import math
import time
import torch
import torch.nn as nn       
import numpy as np
from rich.console import Console
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch.nn.functional as F
from utils.pepler_dataloader import DataLoader_Rs
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader,Dataset
import itertools
from utils.pepler_dataloader import Dataset_Rs_Pytorch
from utils.lr_utils import WarmUpLR
from rich.progress import BarColumn, Progress
from rich.live import Live
from rich.console import Console
from tqdm import tqdm
import datetime

console = Console()


'''
    Encoder: map user-item pair into latent space
'''
class Encoder(nn.Module):
    def __init__(self,n_user,n_item,embedding_size,latent_dim):
        super(Encoder,self).__init__()
        self.user_embeddings = nn.Embedding(n_user, embedding_size)
        self.item_embeddings = nn.Embedding(n_item, embedding_size)

        self.user_embeddings.weight.data.uniform_(-0.2, 0.2)
        self.item_embeddings.weight.data.uniform_(-0.2, 0.2)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model = embedding_size * 2, nhead = 4, dim_feedforward = 512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = 2)

        # mu and sigma
        self.mu_l = nn.Linear(embedding_size * 2, latent_dim)
        self.logvar_l = nn.Linear(embedding_size * 2, latent_dim)

    def forward(self, user, item):
        user_embedding = self.user_embeddings(user)
        item_embedding = self.item_embeddings(item)
        ui_embedding = torch.cat([user_embedding, item_embedding], dim=1)
        ui_out = self.transformer_encoder(ui_embedding)
        mu = self.mu_l(ui_out)
        log_sigma2 = self.logvar_l(ui_out)
        return mu,log_sigma2
'''
    Decoder: decoder the rating
'''
class Decoder(nn.Module):
    def __init__(self,latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),

            nn.Linear(256, 5))
    def forward(self,z):
        z = self.fc1(z)
        return z


class Vae_Cluster_Es(nn.Module):
    '''
        args: 
            n_user: number of user
            n_item: number of item
            args  : parser parameter
    '''
    def __init__(self, n_user, n_item, args):
        super(Vae_Cluster_Es, self).__init__()

        self.encoder      = Encoder(n_user = n_user, n_item = n_item, embedding_size=args.embedding_size, latent_dim=args.latent_dim)
        self.decoder      = Decoder(latent_dim = args.latent_dim)
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.phi          = nn.Parameter(torch.FloatTensor(args.num_cluster,).fill_(1) / args.num_cluster, requires_grad=True)
        self.mu_c         = nn.Parameter(torch.FloatTensor(args.num_cluster, args.latent_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(args.num_cluster, args.latent_dim).fill_(0), requires_grad=True)
        self.args         = args


    def now_time(self):
        return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '
    def plot_loss_curve(self, losses, title='Pretraining Loss Curve', save_path='loss_curve.png', dpi=200):
        plt.figure(figsize=(10, 5))

        sns.lineplot(x = range(len(losses)), y = losses,marker='*',markerfacecolor='#F0988C', markersize=16, markevery=10)
        plt.gca().lines[0].set_color('#C76DA2')
        plt.gca().lines[0].set_linestyle('-')
        plt.gca().lines[0].set_linewidth(2.5)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True)
        plt.savefig(save_path, dpi=dpi)
        plt.close()

    # 先用预训练把latent space训练好 得到先验分布
    def pretrain(self, corpus, pretrain_epoch):
        assert self.args.pretrain_weight_save is not None
        print(f'Start Pretraining !!!!!')
        if not os.path.exists(os.path.join(os.path.join(self.args.pretrain_weight_save,self.args.dataset, self.args.dataset + f'_cluster_{self.args.num_cluster}_'  +'pretrain_weight.pth'))):
            warm = 1
            Loss = nn.CrossEntropyLoss()        
            optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()),lr = 0.00015)# 0.00015
            train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [60, 120], gamma = 0.9) #learning rate decay
            # batch_size influences the Training Time in V100 32GB. ex: 81920 vs 1024, meanwhile, which influences the result of clustering
            # btw: using Adam optimizer, remember to set small batch size like 256,512
            data_loader = DataLoader(Dataset_Rs_Pytorch(corpus.train),batch_size = 2048,shuffle = True)
            iter_per_epoch = len(data_loader)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
            print("================================================ Pretraining Start ================================================")
            epoch_bar = tqdm(range(pretrain_epoch))
            start = time.time()
            losses = []
            
            best_val_loss = float('inf')
            endure_count = 0
            endure_count_break = 15
            for epoch in epoch_bar:
                total_sample = 0. 
                losses_epoch = 0.
                epoch = epoch + 1
                if epoch > warm:
                    train_scheduler.step(epoch)
                for batch_index,(user, item, rating, _, _) in enumerate(data_loader):
                    user = user.to(self.device)
                    item = item.to(self.device)
                    rating = rating - 1
                    rating = rating.to(self.device)
                    optimizer.zero_grad()
                    mu,log = self.encoder(user,item)
                    pre_rating = self.decoder(mu)
                    loss = Loss(pre_rating, rating)
                    losses_epoch += loss
                    loss.backward()
                    optimizer.step()
                    if batch_index % 100 == 0:
                        losses.append(loss.item())
                    console.print(':thumbs_up: :pile_of_poo: Time:{time} Training Epoch: {epoch}/{all_epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                            loss.item(),
                            optimizer.param_groups[0]['lr'],
                            time = self.now_time(),
                            epoch=epoch,
                            all_epoch=pretrain_epoch,
                            trained_samples=batch_index * data_loader.batch_size + len(user),
                            total_samples=len(data_loader.dataset)
                    ), style="bold red")
                    total_sample += data_loader.batch_size
            
                if epoch <= warm:
                    warmup_scheduler.step()
                losses_epoch = losses_epoch / len(data_loader)
                if losses_epoch < best_val_loss:      
                    best_val_loss = losses_epoch
                    torch.save(self.state_dict(), os.path.join(self.args.pretrain_weight_save,self.args.dataset,self.args.dataset + f'_cluster_{self.args.num_cluster}_' + 'best_' +'pretrain_weight.pth'))
                    print(f'Saving Best Pretraining Model for loss {best_val_loss}')
                else:
                    endure_count += 1
                    console.print(self.now_time() + 'We are going to early stop..., Which is Harder...')
                    if endure_count == endure_count_break:
                        break
             
            finish = time.time() 
            print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
            self.encoder.logvar_l.load_state_dict(self.encoder.mu_l.state_dict())
            self.plot_loss_curve(losses, title='Pretrain Loss Curve',save_path=os.path.join(self.args.pretrain_weight_save,self.args.dataset,self.args.dataset + '_' + 'pretrain_loss.png'))
            
            Z = []
            with torch.no_grad():
                for user, item, rating, _, _ in data_loader:
                    user = user.to(self.device)
                    item = item.to(self.device)
                    z1, z2 = self.encoder(user,item)
                    assert F.mse_loss(z1, z2) == 0
                    Z.append(z1)
            # Z shape : batch,latent_dim
            Z = torch.cat(Z, 0).detach().cpu().numpy()


            gmm = GaussianMixture(n_components = self.args.num_cluster, covariance_type='diag')
            pre = gmm.fit_predict(Z)

            num_samples_per_cluster = 500
            indices = []
            for i in range(self.args.num_cluster):
                indices_in_cluster = np.where(pre == i)[0]
                selected_indices = np.random.choice(indices_in_cluster, num_samples_per_cluster, replace=False)
                indices.extend(selected_indices)

            selected_Z = Z[indices]

            tsne = TSNE(n_components=2, random_state=42)  
            Z_2d = tsne.fit_transform(selected_Z)
            selected_pre = pre[indices]

            plt.figure(figsize=(10, 8))
            plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c = selected_pre, cmap='viridis', alpha=0.6)

            plt.colorbar()  
            plt.title(f'Vis of Pretrain Latent Space for {self.args.dataset}') 
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.savefig(os.path.join(self.args.pretrain_weight_save,self.args.dataset, self.args.dataset + '_' + f'pretrain_latent_{self.args.num_cluster}'),dpi=300)

            print('GaussianMixture Model Fit Done......')
            self.phi.data = torch.from_numpy(gmm.weights_).cuda().float()
            self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
            # 注意这里取了log
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())

            if not self.args.pretrain_weight_save:
                os.mkdirs(self.args.pretrain_weight_save)

            torch.save(self.state_dict(), os.path.join(self.args.pretrain_weight_save,self.args.dataset,self.args.dataset + f'_cluster_{self.args.num_cluster}_'  +'pretrain_weight.pth'))
            self.load_state_dict(torch.load(os.path.join(self.args.pretrain_weight_save,self.args.dataset,self.args.dataset + f'_cluster_{self.args.num_cluster}_'  +'pretrain_weight.pth')))
            print('loaded best pretrain weight')
        else:
            self.load_state_dict(torch.load(os.path.join(self.args.pretrain_weight_save,self.args.dataset, self.args.dataset + f'_cluster_{self.args.num_cluster}_'  +'pretrain_weight.pth')))
            print('already loaded')

    def Elbo_loss(self, user, item, rating, scale_factor_kl):
        # parameters: L = 5, KL regu: 0.1
        L                  = 1 
        z_mu, z_sigma2_log = self.encoder(user, item)
        loss_func          = nn.CrossEntropyLoss()
        # start sampling   
        elbo_loss          = 0
        for l in range(L): 
            # reparameterization trick
            z = torch.randn_like(z_mu) * torch.exp(0.5 * z_sigma2_log) + z_mu
            z = z.to(self.device)
            pre_rating = self.decoder(z)
            elbo_loss += loss_func(pre_rating, rating)


        Loss           = elbo_loss /  L * self.args.embedding_size * 2

        pi=self.phi
        log_sigma2_c=self.log_sigma2_c
        mu_c=self.mu_c
        # resampling
        z = torch.randn_like(z_mu) * torch.exp(0.5 * z_sigma2_log) + z_mu
        
        z = z.to(self.device)

        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+1e-10

        yita_c = yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters

        kl = 0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))
        kl -= torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1)) + 0.5*torch.mean(torch.sum(1+z_sigma2_log,1))
        kl = kl * scale_factor_kl
        Loss += kl
        return Loss * 0.1

    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]  
        for c in range(self.args.num_cluster): 
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))

        return torch.cat(G,1)

    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))




    # TODO: Predict User-Item Cluster
    '''
        ref: gmm pdf calculate
        predict_cluster_index: predict user-item-pair clsuter index && predict moe gate index
        args: 
            user: (batch,) 
            item: (batch,)
        return 
            cluster_index: int
    '''
    def predict_cluster_index(self, user, item):
        # get the mu & log_sigma2
        z_mu, z_sigma2_log = self.encoder(user, item)

        z_mu = z_mu.detach()
        z_sigma2_log = z_sigma2_log.detach()
        z = torch.randn_like(z_mu) * torch.exp(0.5 * z_sigma2_log) + z_mu
        z = z.detach()
        gmm_pai        = self.phi
        gmm_mu_c       = self.mu_c
        gmm_log_sigma2 = self.log_sigma2_c


        yita = torch.argmax(torch.exp(torch.log(gmm_pai.unsqueeze(0))) + self.gaussian_pdfs_log(z,gmm_mu_c,gmm_log_sigma2),dim=1)
        return yita.detach()# .numpy()
