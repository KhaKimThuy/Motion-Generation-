%%writefile /kaggle/working/motion-diffusion-model/model/mdm.py
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
from model.qna import FusedQnA1d
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
            use_qna=False,
            kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.use_qna = use_qna
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        if not use_qna:
            self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = FusedQnA1d(
            in_features=self.channels,
            timesteps_features=None,
            hidden_features=self.channels,
            heads=self.num_heads,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        h = x.reshape(b, c, 1, -1)
        h = self.attention(h)
        h = h.reshape(b, c, -1)
        return h.reshape(b, c, *spatial)

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

# Custom Multi-Head Attention
class OrinMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear layers for query, key, and value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T, D = query.size()
        H = self.num_heads

        # Linear projections and split into heads
        Q = self.q_proj(query).view(B, T, H, -1).transpose(1, 2)  # (B, H, T, head_dim)
        K = self.k_proj(key).view(B, T, H, -1).transpose(1, 2)
        V = self.v_proj(value).view(B, T, H, -1).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        context = torch.matmul(attn, V)  # (B, H, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        # Final projection
        output = self.out_proj(context)
        return output

class EfTemporalAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        x:  T, B, D
        """
        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)


        B, T, D = q.shape
        H = self.num_heads
        # B, T, D
        query = self.query(self.norm(q))
        # B, T, D
        # key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        key = self.key(self.norm(k))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        # value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        value = (self.value(self.norm(v))).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = self.proj_out(y) # b, t, d
        return y.permute(1,0,2)

class SpatialMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        :param size: Kích thước (dimension) của các vector truy vấn (query), key và value
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

        # self.kv_layer = AttentionBlock(
        #                         d_model,
        #                         use_checkpoint=False,
        #                         num_heads=num_heads,
        #                         num_head_channels=-1,
        #                         use_new_attention_order=False,
        #                         use_qna=True,
        #                         kernel_size=3,
        #                 )

        self.kv_layer = nn.Conv1d(d_model, num_heads * head_size, kernel_size=3, stride=1)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.output_layer =  nn.Linear(d_model, d_model)


    def forward(self, q, k, v, mask=None, key_padding_mask=None):

        batch_size, t, d = q.shape
        num_heads = self.num_heads

        q = self.q_layer(q)
        kv = self.kv_layer(k.permute(0,2,1)).permute(0,2,1)
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
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)

        elif self.arch == "lgtt":
            print("********* ME MODEL *********")
            self.seqTransEncoder = CustomTransformerEncoder(d_model=self.latent_dim, \
                                                num_heads=self.num_heads, \
                                                num_layers=self.num_layers, \
                                                dropout=self.dropout, \
                                                dim_feedforward=self.ff_size, \
                                                activation=activation)

        elif self.arch == 'mamba':

            from mamba_ssm import Mamba

            print("MAMBA init")
            self.seqMamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=self.latent_dim, # Model dimension d_model
            d_state=64,  # SSM state expansion factor
            d_conv=64,    # Local convolution width
            expand=16,    # Block expansion factor
        ).to("cuda")
        

        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

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
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        

        if self.arch == 'lgtt':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)#.permute(1,0,2)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]#.permute(1,0,2)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        


        elif self.arch == 'mamba':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            # print(xseq.shape)
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            # print(xseq.shape)
            xseq = xseq.permute(1,0,2)
            # print(xseq.shape) # torch.Size([4, 329, 512])
            output = self.seqMamba(xseq)[:,1:,:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            # print(output.shape) torch.Size([4, 328, 512])
            output = output.permute(1,0,2)
            # print(output.shape)  torch.Size([328, 4, 512])

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

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
