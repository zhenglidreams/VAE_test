import sys

if '../' not in sys.path:
    sys.path.append('../')

from operator import mul
from functools import reduce
from collections import OrderedDict
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.data.dataloader import _InfiniteConstantSampler
import pytorch_lightning as pl
import settings.settings as stgs
from grammar_vae.SentenceGenerator import SentenceGenerator
from grammar_vae.nas_grammar import grammar
import numpy as np





from IPython import embed

class Unflatten(nn.Module):
    def __init__(self, size):
        """
        :param size: tuple (channels, length)
        """
        super().__init__()
        self.h, self.w = size

    def forward(self, x):
        """
        :param x: 2d-Tensor of dimensions (n_batches, self.h * self.w)
        """
        n_batches = x.size(0)
        assert self.h * self.w == x.shape[-1]
        return x.view(n_batches, self.h, self.w)


def compute_dimensions(input_sz, module):
    with torch.no_grad():
        x = torch.ones(1, *input_sz, dtype=torch.float)
        size = tuple(module(x).shape)
    return size


class joint_vae_encoder(nn.Module):
    def __init__(self, ch_in, hids, ks, ss, ps):
        super().__init__()
        en_modules=[]
        for i in range(len(hids)):
            en_modules.append(
                nn.Sequential(
                nn.Conv1d(ch_in, out_channels=hids[i],
                          kernel_size=ks[i], stride=ss[i], padding=ps[i]),
                nn.BatchNorm1d(hids[i]),
                nn.ReLU()))
            ch_in=hids[i]
        self.layers=nn.Sequential(*en_modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class joint_vae_decoder(nn.Module):
    def __init__(self, ch_in, hids, ks, ss, ps,ops):
        super().__init__()
        modules = []
        for i in range(len(hids) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hids[i],
                                       hids[i + 1],
                                       kernel_size=ks[i],
                                       stride=ss[i],
                                       padding=ps[i],
                                       output_padding=ops[i]
                                       ),
                    nn.BatchNorm1d(hids[i + 1]),
                    nn.ReLU())
            )

        modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hids[-1],
                                       hids[-1],
                                       kernel_size=ks[-1],
                                       stride=ss[-1],
                                       padding=ps[-1],

                                       output_padding=ops[-1]
                                       ),
                    nn.BatchNorm1d(hids[-1]),
                    nn.ReLU(),
                    nn.Conv1d(hids[-1], ch_in,
                              kernel_size=2, padding=0),
                    nn.Tanh())
        )
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class NA_VAE(pl.LightningModule):

    def __init__(self, hparams=stgs.VAE_HPARAMS):

        super().__init__()
        self.hparams = Namespace(**hparams)
        self.hp = hparams  # alias
        # self.device = self.hp['device']
        self.n_chars = self.hp['n_chars']
        self.bsz = self.hp['batch_size']
        # self.data_sz = self.hp['data_size']
        self.max_len = self.hp['max_len']
        self.conv_channels = [int(s) for s in self.hp['channels'].split(',')]
        self.conv_k_sizes = [int(s) for s in self.hp['k_sizes'].split(',')]
        self.conv_strides = [int(s) for s in self.hp['strides'].split(',')]
        self.fc_dim = self.hp['fc_dim']
        self.latent_sz = self.hp['latent_size']
        self.rnn_hidden = self.hp['rnn_hidden']
        self.drop_rate = self.hp['drop_rate']
        self.data_gen = None
        self.grammar = grammar
        self.ind_of_ind = torch.tensor(grammar.ind_of_ind, device='cuda')
        self.masks = torch.tensor(grammar.masks, device='cuda')
        self.kld_weight = 0.
        self.iter = 0.
        self.alpha = self.hp["alpha"]
        self.temp = self.hp["temperature"]
        self.min_temp = self.hp["temperature"]
        self.categorical = self.hp['categorical_channel']
        self.joint_hidden=self.hp['joint_hidden']
        self.joint_k_sizes=self.hp['joint_k_sizes']
        self.joint_strides=self.hp['joint_strides']
        self.joint_padding=self.hp['joint_padding']
        self.joint_hidden_d = self.hp['joint_hidden_d']
        self.joint_k_sizes_d = self.hp['joint_k_sizes_d']
        self.joint_strides_d = self.hp['joint_strides_d']
        self.joint_padding_d = self.hp['joint_padding_d']
        self.joint_outpadding_d = self.hp['joint_outpadding_d']
        self.cont_min = self.hp['con_min_capacity']
        self.cont_max = self.hp['con_max_capacity']
        self.disc_min = self.hp['dis_min_capacity']
        self.disc_max = self.hp['dis_max_capacity']
        self.cont_gamma = self.hp['cont_gamma']
        self.disc_gamma = self.hp['disc_gamma']
        self.cont_iter = self.hp['cont_iter']
        self.disc_iter = self.hp['disc_iter']
        self.anneal_rate = self.hp['anneal_rate']
        self.anneal_interval =self.hp['anneal_interval']
        self.joint_encoder = joint_vae_encoder(self.n_chars, self.joint_hidden, self.joint_k_sizes, self.joint_strides,
                                         self.joint_padding)
        #_, self.H, self.W = compute_dimensions((self.n_chars, self.max_len), self.beta_encoder)

        self.joint_decoder=joint_vae_decoder(self.n_chars, self.joint_hidden_d, self.joint_k_sizes_d, self.joint_strides_d,
                                         self.joint_padding_d,self.joint_outpadding_d)

        self.mu = nn.Linear(self.joint_hidden[-1] * 4, self.latent_sz)
        self.logvar = nn.Linear(self.joint_hidden[-1] * 4, self.latent_sz)
        self.q_fc = nn.Linear(self.joint_hidden[-1] * 4, self.categorical)
        self.decode_input=nn.Linear(self.latent_sz+self.categorical,self.joint_hidden[-1]*4)

    def add_latent_vectors(self, hot_vector):
        if self.hot_list_started == False:
            self.hot_list[0] = hot_vector
            self.hot_list_started = True
        else:
            self.hot_list.append(hot_vector)

    def send_latent_vectors(self):
        #  self.latent_vector = torch.cat((self.latent_vector, hot_vector), 1)  # appends hot vector to the batch
        self.latent_vector = torch.stack(self.hot_list, 0)  # appends hot vector to the batch
        self.hot_list_started = False
        return self.latent_vector  # Sends the latent vectors to the predictor

    def encode(self, x):
        h = self.joint_encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu_ = self.mu(h)
        logvar_ = self.logvar(h)
        z = self.q_fc(h)
        return mu_, logvar_,z

    def decode(self, z):
        h=self.decode_input(z)
        h=h.view(-1,self.joint_hidden[-1],4)
        decode_ouput=self.joint_decoder(h)
        return decode_ouput

    def forward(self, x):
        mu, logvar,q = self.encode(x.squeeze())
        z = self.reparameterize(mu, logvar,q)
        #z = torch.cat((z, lengths.transpose(0, 1)), dim=1)  # add length information to latent vector
        x_recon = self.decode(z)
        return x_recon, mu, logvar,q
        #是decode后  还原 原来值的结果 当然 跟原来值会有差距

    def reparameterize(self, mu, logvar,q):
        
        std = torch.exp(0.5 * logvar)
        e= torch.randn_like(std)
        z = e * std + mu
        u = torch.rand_like(q)
        eps = 1e-7
        g = - torch.log(- torch.log(u + eps) + eps)
        # Gumbel-Softmax sample
        s = F.softmax((q + g) / self.temp, dim=-1)
        s = s.view(-1, self.categorical)
        return torch.cat([z, s], dim=1)

    def conditional(self, x_true, x_pred):
        most_likely = torch.argmax(x_true, dim=1)
        lhs_indices = torch.index_select(self.ind_of_ind, 0, most_likely.view(-1)).long()
        M2 = torch.index_select(self.masks, 0, lhs_indices).float()
        M3 = M2.reshape((-1, self.max_len, self.n_chars))
        M4 = M3.permute((0, 2, 1))
        P2 = torch.exp(x_pred) * M4
        # P2 = torch.exp(x_pred)
        P2 = P2 / (P2.sum(dim=1, keepdim=True) + 1.e-10)
        return P2

    def loss_function(self, x_decoded_mean, x_true, mu, logvar,q,batch_id_x):
        x_cond = self.conditional(x_true.squeeze(), x_decoded_mean)
        x_true = x_true.view(-1, self.n_chars * self.max_len)
        x_cond = x_cond.view(-1, self.n_chars * self.max_len)
       
        assert x_cond.shape == x_true.shape
        bce = F.binary_cross_entropy(x_cond, x_true, reduction='sum')
        if batch_id_x % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_id_x),
                                   self.min_temp)
        
        q_p = F.softmax(q, dim=-1)
        disc_curr = (self.disc_max - self.disc_min) * \
                    self.iter / float(self.disc_iter) + self.disc_min
        disc_curr = min(disc_curr, np.log(self.categorical))
        eps = 1e-7
        h1 = q_p * torch.log(q_p + eps)
        h2 = q_p * np.log(1. / self.categorical + eps)
        kl_disc_loss = torch.mean(torch.sum(h1 - h2, dim=1), dim=0)
        cont_curr = (self.cont_max - self.cont_min) * \
                    self.iter/ float(self.cont_iter) + self.cont_min
        cont_curr = min(cont_curr, self.cont_max)
        kl_div = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        capacity_loss = self.disc_gamma * torch.abs(disc_curr - kl_disc_loss) + \
                        self.cont_gamma * torch.abs(cont_curr - kl_div)


        return self.alpha * bce + self.kld_weight * capacity_loss, bce, capacity_loss

    @staticmethod
    def check_sz(x, target_size):
        assert x.shape[1:] == target_size, f'Wrong spatial dimensions: {tuple(x.shape[1:])}; ' \
            f'expected {target_size}'

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters(), lr=self.hp['lr_ini'], eps=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min',
                                                                    factor=self.hp['lr_reduce_factor'],
                                                                    patience=self.hp['lr_reduce_patience'],
                                                                    min_lr=self.hp['lr_min'])
        return [self.optim], [self.scheduler]

    def training_step(self, batch, batch_idx):
        # Set up KL Divergence annealing
        t0, t1 = self.hp['kld_wt_zero_for_epochs'], self.hp['kld_wt_one_at_epoch']  # for kld annealing
        if self.iter < self.hp['kld_wt_zero_for_epochs']:
            self.kld_weight = self.hp['kld_zero_value']
        else:
            self.kld_weight = min(self.hp['kld_full_value'] * (self.iter - t0) / (t1 - t0),
                                  self.hp['kld_full_value'])

        x, lengths = batch
        recon_x, mu, logvar,q = self.forward(x)
        loss_val, bce, kl_div = self.loss_function(recon_x,x, mu, logvar,q,batch_idx)
        lr = torch.tensor(self.optim.param_groups[0]['lr'])
        progress_bar = OrderedDict({'bce': bce.mean(), 'kl_div': kl_div.mean(),
                                    'lr': lr,
                                    'iter': self.iter,
                                    'kld_weight': self.kld_weight})
        log = OrderedDict({'loss': loss_val.mean(dim=0), 'bce': bce.mean(dim=0), 'kl_div': kl_div.mean(dim=0),
                           'lr': lr,
                           'kld_weight':
            self.kld_weight})
        self.iter += 1
        return {'loss': loss_val,
                'bce': bce,
                'kl_div': kl_div,
                'kld_weight': self.kld_weight,
                'iter': self.iter,
                'lr': lr,
                'log': log,
                'progress_bar': progress_bar}


    def test_step(self, batch, batch_idx):
        if self.on_gpu:
            x = batch.cuda()
        else:
            x = batch
        x = batch
        recon_x, mu, logvar = self.forward(x)
        loss_val, _, _ = self.loss_function(recon_x, x, mu, logvar,batch_idx)
        return {'test_loss': loss_val.mean(dim=0)}

    def training_end(self, outputs):
        if not isinstance(outputs, list):
            outputs = [outputs]
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        bce_mean = torch.stack([x['bce'] for x in outputs]).mean()
        kld_mean = torch.stack([x['kl_div'] for x in outputs]).mean()
        kld_weight = outputs[-1]['kld_weight']
        lr = outputs[-1]['lr']
        progress_bar = OrderedDict({'bce_mean': bce_mean, 'kld_mean': kld_mean,
                                    'kld_weight': kld_weight, 'lr': lr,
                                    'iter': self.iter})
        log = OrderedDict({'loss': loss_mean, 'bce_mean': bce_mean, 'kld_mean': kld_mean,
                                    'kld_weight': kld_weight, 'lr': lr})
        return {'loss': loss_mean,
                'val_loss': loss_mean,
                'log': log,
                'progress_bar': progress_bar
                }

    def test_epoch_end(self, batch, batch_idx):
        if not isinstance(outputs, list):
            outputs = [outputs]
        loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        log = OrderedDict({'test_loss': loss_mean})
        return {'test_loss': loss_mean, 'log': log}

    def get_data_generator(self, min_sample_len, max_sample_len, seed=0):
        self.data_gen = SentenceGenerator(grammar.GCFG, min_sample_len, max_sample_len, self.bsz, seed)

    def train_dataloader(self):
        self.ldr = DataLoader(self.data_gen, batch_size=1, num_workers=0)
        return self.ldr

    def test_dataloader(self):
        return DataLoader(self.data_gen, batch_size=1, num_workers=0)

if __name__ == '__main__':
    from torch.utils.data import IterableDataset

    vae = NA_VAE(len(grammar.GCFG.productions()))
    #vae.get_data_generator(3, 30)
    #trainer = pl.Trainer(min_nb_epochs=100, val_check_interval=100, early_stop_callback=None)
    #trainer.fit(vae)

    vae.get_batch()
    print(vae.grammar_str)

    mu1 = torch.rand(1, 5)
    mu2 = torch.rand(1, 5)
    mu3 = torch.rand(1, 5)
    print(mu1)

    vae.add_latent_vectors(mu1)
    vae.add_latent_vectors(mu2)
    vae.add_latent_vectors(mu3)
    print(vae.hot_list)
    print("_________")
    print(vae.send_latent_vectors())
