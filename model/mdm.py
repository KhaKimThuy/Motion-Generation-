# %%writefile /kaggle/working/c_mdm/model/mdm.py
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz

from diffusion.nn import (
    checkpoint,
    conv_nd
)


class SpatialMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        :param size: KÃ­ch thÆ°á»›c (dimension) cá»§a cÃ¡c vector truy váº¥n (query), key vÃ  value
        """
        super().__init__()

        assert d_model % num_heads == 0

        print(f"Dropout in TEMP : {dropout}")

        self.head_size = head_size = d_model // num_heads
        self.model_size = d_model
        self.num_heads = num_heads

        self.q_layer = nn.Linear(d_model, num_heads * head_size)
        self.k_layer = nn.Linear(d_model, num_heads * head_size)
        self.v_layer = nn.Linear(d_model, num_heads * head_size)

        self.norm = nn.LayerNorm(d_model)

        # self.kv_layer = AttentionBlock(
        #                         d_model,
        #                         use_checkpoint=False,
        #                         num_heads=num_heads,
        #                         num_head_channels=-1,
        #                         use_new_attention_order=False,
        #                         use_qna=True,
        #                         kernel_size=3,
        #                     )
        self.kv_layer = nn.Conv1d(d_model, num_heads * head_size, kernel_size=3, stride=1)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.output_layer =  nn.Linear(d_model, d_model)


    def forward(self, q, k, v, mask=None, key_padding_mask=None):

        batch_size, t, d = q.shape
        num_heads = self.num_heads

        q = self.q_layer(q)
        kv = self.norm(self.kv_layer(k.permute(0,2,1)).permute(0,2,1))
        # k, v = torch.chunk(kv, 2, dim=-1)
        k = self.k_layer(kv)
        v = self.v_layer(kv)
        # k = self.k_layer(k.permute(0,2,1)).permute(0,2,1)
        # v = self.v_layer(v.permute(0,2,1)).permute(0,2,1)

        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size)

        context = self.output_layer(context)

        # return context.permute(1,0,2), attention
        return context, attention

# Transformer Encoder Layer
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, activation='relu'):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        # Multi-Head Attention
        self.temp_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        # self.self_attn = EfTemporalAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        # self.temp_attn = TemporalMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.spat_attn = SpatialMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        # Feedforward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer Normalization
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass through a custom Transformer Encoder Layer.

        Args:
            src (Tensor): Input tensor of shape (sequence_length, batch_size, d_model).
            src_mask (Tensor, optional): Mask for the source sequence (optional).
            src_key_padding_mask (Tensor, optional): Mask for padding in the batch (optional).

        Returns:
            Tensor: Output tensor of shape (sequence_length, batch_size, d_model).
        """
        src_norm = self.norm0(src)
        
        # Self-attention with residual connection and normalization
        attn_output_1, _ = self.temp_attn(
            query=src_norm,
            key=src_norm,
            value=src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        # src = self.dropout(attn_output) + src  # Residual connection
        # src_norm = self.norm1(src)  # Layer normalization
        
        attn_output_2, _ = self.spat_attn(
            q=src_norm,
            k=src_norm,
            v=src_norm,
            mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = self.dropout(attn_output_1) * self.dropout(attn_output_2) + src
        src_norm = self.norm2(src)  # Layer normalization

        # Feedforward network with residual connection and normalization
        ffn_output = self.ffn(src_norm)
        src = src + ffn_output  # Residual connection

        return src

# Transformer Encoder
class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dim_feedforward, dropout, activation):
        super(CustomTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers


    # def __init__(self, d_model, num_heads, dim_feedforward, dropout, activation='relu'):
    #     super(CustomTransformerEncoderLayer, self).__init__()

        # Define the layers directly
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation
            )
            for _ in range(self.num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass through the custom Transformer Encoder.

        Args:
            src (Tensor): Input tensor of shape (sequence_length, batch_size, d_model).
            src_mask (Tensor, optional): Mask for the source sequence (optional).
            src_key_padding_mask (Tensor, optional): Mask for padding in the batch (optional).

        Returns:
            Tensor: Output tensor of shape (sequence_length, batch_size, d_model).
        """
        output = src.permute(1,0,2)
        for layer in self.encoder_layers:
            output = layer(
                src=output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )
        return output.permute(1,0,2)


class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)

        elif self.arch == "lgtt":
            print("LGTT init")
            self.seqTransEncoder = CustomTransformerEncoder(d_model=self.latent_dim, \
                                                num_heads=self.num_heads, \
                                                num_layers=self.num_layers, \
                                                dropout=self.dropout, \
                                                dim_feedforward=self.ff_size, \
                                                activation=activation)



        else:
            raise ValueError('Please choose correct architecture [trans_enc, lgtt]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        print('EMBED TEXT')
        print('Loading CLIP...')
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)


        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape

        # print(f"pose shape ------- : {x.shape}")
        # [10, 263, 1, 196]


        emb = self.embed_timestep(timesteps)  # [1, bs, d]


        force_mask = y.get('uncond', False)
        if 'text_embed' in y.keys():  # caching option
            enc_text = y['text_embed']
        else:
            enc_text = self.encode_text(y['text'])
        emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))


        x = self.input_process(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output
    

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
