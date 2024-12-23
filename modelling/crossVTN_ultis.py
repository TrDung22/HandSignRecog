import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34, resnet50
import torchvision.models as models
import torchvision
from AAGCN.aagcn import AAGCN
from pytorch_lightning.utilities.migration import pl_legacy_patch


class FeatureExtractor(nn.Module):
    """Feature extractor for RGB clips, powered by a 2D CNN backbone."""

    def __init__(self, cnn='rn34', embed_size=512, freeze_layers=0):
        """Initialize the feature extractor with given CNN backbone and desired feature size."""
        super().__init__()

        if cnn == 'rn18':
            model = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif cnn == 'rn34':
            model = resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif cnn == 'rn50':
            model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f'Unknown value for `cnn`: {cnn}')

        if cnn == 'rn18' or cnn == 'rn34' or cnn == 'rn50':
            self.resnet = nn.Sequential(*list(model.children())[:-2])

        # Freeze layers if requested.
        for layer_index in range(freeze_layers):
            for param in self.resnet[layer_index].parameters(True):
                param.requires_grad = False

        # ResNet-18 and ResNet-34 output 512 features after pooling.
        if embed_size != 512 and (cnn == 'rn18' or cnn == 'rn34'):
            self.pointwise_conv = nn.Conv2d(512, embed_size, 1)
        else:
            self.pointwise_conv = nn.Identity()

        if cnn == 'rn50':
            self.pointwise_conv = nn.Conv2d(2048, embed_size, 1)

        self.avg_pool = F.adaptive_avg_pool2d

    def forward(self, rgb_clip):
        """Extract features from the RGB images."""
        b, t, c, h, w = rgb_clip.size()
        # Process all sequential data in parallel as a large mini-batch.
        rgb_clip = rgb_clip.view(b * t, c, h, w)

        features = self.resnet(rgb_clip)
        # Transform to the desired embedding size.
        features = self.pointwise_conv(features)

        # Transform the output of the ResNet (C x H x W) to a single feature vector using pooling.
        features = self.avg_pool(features, 1).squeeze()
        # Restore the original dimensions of the tensor.
        features = features.view(b, t, -1)
        return features

class StreamCrossAttention(nn.Module):
    """Process sequences using self attention."""

    def __init__(self, input_size, hidden_size, n_heads, sequence_size, inner_hidden_factor=2, layer_norm=True,
                 dropout=0.1):
        super().__init__()

        input_sizes = [hidden_size] * len(n_heads)
        input_sizes[0] = input_size
        hidden_sizes = [hidden_size] * len(n_heads)

        self.position_encoding = PositionEncoding(sequence_size, hidden_size)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'stream1': StreamAttentionBlock(
                    inp_size, hid_size, hid_size * inner_hidden_factor,
                    n_head, dropout=dropout, layer_norm=layer_norm
                ),
                'stream2': StreamAttentionBlock(
                    inp_size, hid_size, hid_size * inner_hidden_factor,
                    n_head, dropout=dropout, layer_norm=layer_norm
                )
            })
            for i, (inp_size, hid_size, n_head) in enumerate(zip(input_sizes, hidden_sizes, n_heads))
        ])

    def forward(self, x1, x2):
        x1 = self.position_encoding(x1)
        x2 = self.position_encoding(x2)
        for layer in self.layers:
            x1 = layer['stream1'](x1, context=x2)
            x2 = layer['stream2'](x2, context=x1)
        return x1, x2

class ViewsCrossAttention(nn.Module):
    """Process sequences using self attention."""

    def __init__(self, input_size, hidden_size, n_heads, sequence_size, inner_hidden_factor=2, layer_norm=True,
                 dropout=0.1):
        super().__init__()

        input_sizes = [hidden_size] * len(n_heads)
        input_sizes[0] = input_size
        hidden_sizes = [hidden_size] * len(n_heads)

        self.position_encoding = PositionEncoding(sequence_size, hidden_size)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'view1': ViewsAttentionBlock(
                    inp_size, hid_size, hid_size * inner_hidden_factor,
                    n_head, dropout=dropout, layer_norm=layer_norm
                ),
                'view2': ViewsAttentionBlock(
                    inp_size, hid_size, hid_size * inner_hidden_factor,
                    n_head, dropout=dropout, layer_norm=layer_norm
                ),
                'view3': ViewsAttentionBlock(
                    inp_size, hid_size, hid_size * inner_hidden_factor,
                    n_head, dropout=dropout, layer_norm=layer_norm
                )
            })
            for i, (inp_size, hid_size, n_head) in enumerate(zip(input_sizes, hidden_sizes, n_heads))
        ])

    def forward(self, x1, x2, x3):
        x1 = self.position_encoding(x1)
        x2 = self.position_encoding(x2)
        x3 = self.position_encoding(x3)
        for layer in self.layers:
            x1 = layer['view1'](x1, context1=x2, context2=x3)
            x2 = layer['view2'](x2, context1=x1, context2=x3)
            x3 = layer['view3'](x3, context1=x1, context2=x2)
        return x1, x2, x3
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, num_classes)

        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        return self.fc(self.dropout(x))


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        # self.softmax = BottleSoftmax()

    def forward(self, q, k, v):
        # q.size(): [nh*b x t x d_k]
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        # attn = self.softmax(attn)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output


class LayerNormalization(nn.Module):
    """ Layer normalization module """

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.shape[1] == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, input_size, output_size, d_k, d_v, dropout=0.1, layer_norm=True):
        """
        Args:
            n_head: Number of attention heads
            input_size: Input feature size
            output_size: Output feature size
            d_k: Feature size for each head
            d_v: Feature size for each head
            dropout: Dropout rate after projection
        """
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, input_size, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, input_size, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, input_size, d_v))

        self.attention = ScaledDotProductAttention(input_size)
        self.layer_norm = LayerNormalization(input_size) if layer_norm else nn.Identity()

        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.w_qs)
        nn.init.xavier_normal_(self.w_ks)
        nn.init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)  # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)  # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)  # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs = self.attention(q_s, k_s, v_s)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        split_size = mb_size.item() if isinstance(mb_size, torch.Tensor) else mb_size
        h, t, e = outputs.size()
        outputs = outputs.view(h // split_size, split_size, t, e)  # (H x B x T x E)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(split_size, len_q, -1)  # (B x T x H*E)

        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual)


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_hid, d_inner_hid, dropout=0.1, layer_norm=True):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise
        self.layer_norm = LayerNormalization(d_hid) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class DecoderBlock(nn.Module):
    """ Compose with two layers """

    def __init__(self, input_size, hidden_size, inner_hidden_size, n_head, d_k, d_v, dropout=0.1, layer_norm=True):
        super(DecoderBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, input_size, hidden_size, d_k, d_v, dropout=dropout,
                                           layer_norm=layer_norm)
    def forward(self, enc_input):
        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input
        )
        return enc_output


class CrossAttentionBlock(nn.Module):
    def __init__(self, input_size, hidden_size, inner_hidden_size, n_heads, dropout, layer_norm=True):
        super().__init__()
        self.cross_attn = MultiHeadAttention(
            n_heads, input_size, hidden_size,
            d_k=hidden_size // n_heads, d_v=hidden_size // n_heads,
            dropout=dropout, layer_norm=layer_norm
        )
        self.feed_forward = PositionwiseFeedForward(
            hidden_size, inner_hidden_size, dropout=dropout, layer_norm=layer_norm
        )

    def forward(self, x, context):
        x = self.cross_attn(q=x, k=context, v=context)
        x = self.feed_forward(x)
        return x


class StreamAttentionBlock(nn.Module):
    def __init__(self, input_size, hidden_size, inner_hidden_size, n_heads, dropout=0.1, layer_norm=True):
        super().__init__()
        # Self-attention block
        self.self_attn = DecoderBlock(
            input_size, hidden_size, inner_hidden_size, n_heads,
            d_k=hidden_size // n_heads, d_v=hidden_size // n_heads,
            dropout=dropout, layer_norm=layer_norm
        )
        # Cross-attention block
        self.cross_attn = CrossAttentionBlock(
            input_size, hidden_size, inner_hidden_size, n_heads,
            dropout=dropout, layer_norm=layer_norm
        )

    def forward(self, x, context):
        x = self.self_attn(x)
        x = self.cross_attn(x, context)
        return x

class ViewsAttentionBlock(nn.Module):
    def __init__(self, input_size, hidden_size, inner_hidden_size, n_heads, dropout=0.1, layer_norm=True):
        super().__init__()
        # Self-attention block
        self.self_attn = DecoderBlock(
            input_size, hidden_size, inner_hidden_size, n_heads,
            d_k=hidden_size // n_heads, d_v=hidden_size // n_heads,
            dropout=dropout, layer_norm=layer_norm
        )
        # Cross-attention block
        self.cross_attn = CrossAttentionBlock(
            input_size, hidden_size, inner_hidden_size, n_heads,
            dropout=dropout, layer_norm=layer_norm
        )

    def forward(self, x, context1, context2):
        x = self.self_attn(x)
        x1 = self.cross_attn(x, context1)
        x2 = self.cross_attn(x, context2)
        return (x1+x2)/2

class PositionEncoding(nn.Module):
    def __init__(self, n_positions, hidden_size):
        super().__init__()
        # self.enc = nn.Embedding(n_positions, hidden_size, padding_idx=0)
        self.enc = nn.Embedding(n_positions, hidden_size)

        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / hidden_size) for j in range(hidden_size)]
            if pos != 0 else np.zeros(hidden_size) for pos in range(n_positions)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        self.enc.weight = torch.nn.Parameter(torch.from_numpy(position_enc).to(self.enc.weight.device, torch.float))

    def forward(self, x, cls_token_encodings=False):
        size = x.size(1)
        if cls_token_encodings:
            size = size - 1
        indeces = torch.arange(0, size).to(self.enc.weight.device, torch.long)
        encodings = self.enc(indeces)
        if cls_token_encodings:
            x = x[:, 1:] + encodings
        else:
            x = x + encodings
        return x
