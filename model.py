import pandas as pd
import numpy as np
import random
import itertools
import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import sys

def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class City_Embedding(torch.nn.Module):
    def __init__(self, config):
        super(City_Embedding, self).__init__()
        
        self.num_city = config['num_city']
        self.num_cluster = config['num_cluster']
        self.embed = torch.nn.Linear(self.num_city, self.num_cluster)
        
    def forward(self, city_id):
        x = self.embed(city_id)
        
        return x


class Feature_Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Feature_Encoder, self).__init__()
        
        num_feature = config['num_feature']
        num_cluster = config['num_cluster']
        d_model = config['d_model']
        
        num_output = len(config['metrics'])
        
        self.feat_embed = torch.nn.Linear(num_feature, d_model)
        
        self.conv1 = torch.nn.Conv1d(num_output, int(d_model / 4), 3)
        self.pool1 = torch.nn.MaxPool1d(3, 1, padding=0)
        self.conv2 = torch.nn.Conv1d(int(d_model / 4), int(d_model / 2), 3)
        self.pool2 = torch.nn.MaxPool1d(3, 1, padding=0)
        self.conv3 = torch.nn.Conv1d(int(d_model / 2), d_model, 3)
        self.pool3 = torch.nn.MaxPool1d(3, 1, padding=0)
        self.norm1 = torch.nn.LayerNorm(d_model)
        
        self.city_embed = torch.nn.Linear(num_cluster, d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        
    def forward(self, input, history, static):
        x = F.leaky_relu(self.feat_embed(input), 0.2)
        
        h = self.pool1(F.leaky_relu(self.conv1(history), 0.2))
        h = self.pool2(F.leaky_relu(self.conv2(h), 0.2))
        h = self.pool3(F.leaky_relu(self.conv3(h), 0.2))
        
        x = x + h.permute(2, 0, 1)
        x = self.norm1(x)  # (L, N, C)
        
        c = torch.diag_embed(self.city_embed(static))  # (N, C, C)
        x = torch.matmul(x[:, :, None, :], c[None,:]).squeeze(dim=2)  # (L, N, 1, C) * (1, N, C, C) --> (L, N, 1, C)
        x = self.norm2(x)  # (L, N, C)
        
        return x

class Feature_Decoder(torch.nn.Module):
    def __init__(self, config):
        super(Feature_Decoder, self).__init__()
        
        num_feature = config['num_feature']
        num_cluster = config['num_cluster']
        d_model = config['d_model']
        
        num_output = len(config['metrics'])
        
        self.feat_embed = torch.nn.Linear(num_feature, d_model)
        
        self.city_embed = torch.nn.Linear(num_cluster, d_model)
        self.norm = torch.nn.LayerNorm(d_model)
        
    def forward(self, input, static):
        x = F.leaky_relu(self.feat_embed(input), 0.2)
        
        c = torch.diag_embed(self.city_embed(static))  # (N, C, C)
        x = torch.matmul(x[:, :, None, :], c[None,:]).squeeze(dim=2)  # (L, N, 1, C) * (1, N, C, C) --> (L, N, 1, C)
        x = self.norm(x)  # (L, N, C)
        
        return x
    

class Positional_Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Positional_Encoder, self).__init__()
        cycles = np.array(config['pos_enc_cycles'])
        num_multiples = config['d_model'] / (2 * len(cycles))
        multiples = np.arange(1, num_multiples + 1)
        cycles = torch.Tensor(np.reshape(np.tile(cycles[:, None], (1, int(num_multiples))) * 
                              multiples[None, :], (-1)))
        
        self.d_model = config['d_model']
        self.register_buffer('cycles', cycles)
        
    def forward(self, time):
        pe = torch.zeros_like(time)[:, :, None].repeat(1, 1, self.d_model)
        pe[:, :, 0::2] = torch.sin((2 * np.pi / self.cycles[None, None, :]) * time[:, :, None])
        pe[:, :, 1::2] = torch.cos((2 * np.pi / self.cycles[None, None, :]) * time[:, :, None])
        
        return pe
    

class Transformer_Encoder_Layer(torch.nn.Module):
    def __init__(self, config, dropout=0.1):
        super(Transformer_Encoder_Layer, self).__init__()
        
        d_model = config['d_model']
        nhead = config['nhead']
        dim_feedforward = config['dim_feedforward']
        
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Transformer_Encoder(torch.nn.Module):
    def __init__(self, config, dropout=0.1):
        super(Transformer_Encoder, self).__init__()
        
        d_model = config['d_model']
        num_layers = config['num_encoder_layers']
        
        encoder_layer = Transformer_Encoder_Layer(config, dropout)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        output = self.norm(output)

        return output


class Decoder(torch.nn.Module):
    def __init__(self, config, dropout=0.1):
        super(Decoder, self).__init__()
        
        d_model = config['d_model']
        num_output = len(config['metrics'])
        
        self.linear1 = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(d_model, num_output)
        
    def forward(self, src):
        out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return out

class QuantileRegressor(torch.nn.Module):
    def __init__(self, config, dropout=0.1):
        super(QuantileRegressor, self).__init__()
        
        d_model = config['d_model']
        num_output = len(config['metrics'])
        
        self.linear1 = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
        self.out_10 = torch.nn.Linear(d_model, num_output)
        self.out_90 = torch.nn.Linear(d_model, num_output)
        
    def forward(self, src):
        x = self.dropout(F.relu(self.linear1(src)))
        out_10 = self.out_10(x)
        out_90 = self.out_90(x)
        
        return out_10, out_90
    
    
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        # self.optimizer.step()
        
    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    

class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        
        dropout = self.config['dropout']
        warmup = self.config['warmup']
        
        self.city_embed = City_Embedding(config).cuda()
        self.feature_encoder = Feature_Encoder(config).cuda()
        self.feature_decoder = Feature_Decoder(config).cuda()
        self.transformer_encoder = Transformer_Encoder(config, dropout).cuda()
        self.positional_encoder = Positional_Encoder(config).cuda()
        self.decoder = Decoder(config, dropout).cuda()
        self.quantile_regressor = QuantileRegressor(config, dropout).cuda()
        
        self.parameters = itertools.chain(self.city_embed.parameters(),
                                          self.feature_encoder.parameters(),
                                          self.feature_decoder.parameters(),
                                          self.transformer_encoder.parameters(), 
                                          self.decoder.parameters(), 
                                          self.quantile_regressor.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, 
                                          lr=0.0,
                                          weight_decay=5e-5)
        
        self.scheduler = NoamOpt(model_size=config['d_model'],
                                 factor=1.0,
                                 warmup=warmup,
                                 optimizer=self.optimizer)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=.6)
        
    def train(self, 
              feat, label, city_id, mask, time):
        batch_size = feat.size()[1]
        n_output = label.size()[2]
        
        metric_index = self.config['metric_index']
        
        self.city_embed.train()
        self.feature_encoder.train()
        self.feature_decoder.train()
        self.transformer_encoder.train()
        self.decoder.train()
        self.quantile_regressor.train()
        
        self.optimizer.zero_grad()

        #fcst_len = random.randint(self.config['forecast_length_lw_range'], 
        #                          self.config['forecast_length_up_range'])
        fcst_len = self.config['forecast_length']
        
        extr_len = self.config['extra_length']
        n_timestep = feat.size()[0] - extr_len
        
        
        observed = label[: -(fcst_len + 1), :, :].permute(1, 2, 0)  # (N, C, L)
        known = feat[extr_len: , :, :]
        time = time[extr_len: , :]
        mask = mask[extr_len: , :]
        static = self.city_embed(city_id)
        
        feat_enc = self.feature_encoder(known[: -fcst_len], observed, static)
        feat_dec = self.feature_decoder(known[-fcst_len: ], static)
        
        feature = torch.cat((feat_enc, feat_dec), dim=0)
        pe = self.positional_encoder(time)

        attn_mask = _generate_square_subsequent_mask(sz=n_timestep).cuda()
        key_padding_mask = torch.transpose(1. - mask, 0, 1).to(dtype=torch.bool)
        
        input = feature + pe
        x = self.transformer_encoder(src=input,
                                     mask=attn_mask, 
                                     src_key_padding_mask=key_padding_mask)
        prediction = self.decoder(x)
        pred_10, pred_90 = self.quantile_regressor(x)
        
        
        abs_error = torch.sum(torch.abs(prediction[-fcst_len: ] - 
                                        label[-fcst_len: ]) * mask[-fcst_len:, :, None]) / \
                    (torch.sum(mask[-fcst_len: ]) * n_output)
        
        
        
        q10_error = torch.sum((0.9 * torch.clamp(pred_10[-fcst_len: ] - label[-fcst_len: ], min=0.0) +
                               0.1 * torch.clamp(label[-fcst_len: ] - pred_10[-fcst_len: ], min=0.0)) * 
                              mask[-fcst_len:, :, None]) / \
                    (torch.sum(mask[-fcst_len: ]) * n_output)
        q90_error = torch.sum((0.1 * torch.clamp(pred_90[-fcst_len: ] - label[-fcst_len: ], min=0.0) +
                               0.9 * torch.clamp(label[-fcst_len: ] - pred_90[-fcst_len: ], min=0.0)) * 
                              mask[-fcst_len:, :, None]) / \
                    (torch.sum(mask[-fcst_len: ]) * n_output)
        
        
        
        
        # impose regularization on sum of periods
        sum_error = torch.abs(torch.sum((prediction[-fcst_len:, :, :] - 
                                         label[-fcst_len:, :, :]) * mask[-fcst_len:, :, None])) / \
                    (torch.sum(mask[-fcst_len:, :]) * n_output)
        
        # impose regularization on finish < strive < call
        reg_error = torch.sum((torch.clamp(prediction[:, :, metric_index['strive_order_cnt']] - 
                                           prediction[:, :, metric_index['total_no_call_order_cnt']], 
                                           min=0.0) + 
                               torch.clamp(prediction[:, :, metric_index['total_finish_order_cnt']] - 
                                           prediction[:, :, metric_index['strive_order_cnt']], 
                                           min=0.0) + 
                               torch.clamp(prediction[:, :, metric_index['online_payed']] - 
                                           prediction[:, :, metric_index['online_time']],
                                           min=0.0)) * mask) / \
                    torch.sum(mask)
        
        # impose non-negative constraint
        reg_error+= torch.sum((torch.clamp(-prediction, min=0.0) * mask[:, :, None]) / \
                              (torch.sum(mask) * n_output))
        
        objective = abs_error + q10_error + q90_error
        
        objective.backward()
        self.optimizer.step()
        
        return (float(abs_error.cpu().data.numpy()),
                float(q10_error.cpu().data.numpy()),
                float(q90_error.cpu().data.numpy()))
    
    def infer(self, 
              feat, label, city_id, mask, time,
              return_loss=False):
        batch_size = feat.size()[1]
        n_output = label.size()[2]
        
        fcst_len = self.config['forecast_length']
        extr_len = self.config['extra_length']
        n_timestep = feat.size()[0] - extr_len
        
        self.city_embed.eval()
        self.feature_encoder.eval()
        self.feature_decoder.eval()
        self.transformer_encoder.eval()
        self.decoder.eval()
        self.quantile_regressor.eval()
        
        observed = label[: -(fcst_len + 1), :, :].permute(1, 2, 0)
        known = feat[extr_len: , :, :]
        time = time[extr_len: , :]
        mask = mask[extr_len: , :]
        static = self.city_embed(city_id)

        feat_enc = self.feature_encoder(known[: -fcst_len], observed, static)
        feat_dec = self.feature_decoder(known[-fcst_len: ], static)
        
        feature = torch.cat((feat_enc, feat_dec), dim=0)
        pe = self.positional_encoder(time)

        attn_mask = _generate_square_subsequent_mask(sz=n_timestep).cuda()
        
        input = feature + pe
        x = self.transformer_encoder(src=input,
                                     mask=attn_mask)
        prediction = self.decoder(x)
        pred_10, pred_90 = self.quantile_regressor(x)
        
        abs_error = torch.mean(torch.abs(prediction[-fcst_len: ] - 
                                         label[-fcst_len: ]))
        
        if return_loss == True:
            return prediction.cpu().data.numpy(),\
                   pred_10.cpu().data.numpy(),\
                   pred_90.cpu().data.numpy(),\
                   float(abs_error.cpu().data.numpy())
        else:
            return prediction.cpu().data.numpy()
    
    def reset_config(self, config):
        self.config = config
