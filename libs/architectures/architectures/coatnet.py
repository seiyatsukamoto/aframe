import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

from ml4gw.nn.resnet.resnet_1d import BasicBlock

def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class HAttention1D(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64,
        block_size = 16,
        pos_emb = None,
        eps = 1e-8,
        **kwargs
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        inner_dim = heads * dim_head

        self.pos_emb = pos_emb
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        b, n, h, device, bsz, eps = *x.shape[:2], self.heads, x.device, self.block_size, self.eps

        # pad sequence length to power of 2
        pad_to_len = 2 ** ceil(log2(n))
        padding = pad_to_len - n

        if padding != 0:
            x = F.pad(x, (0, 0, 0, padding), value = 0.)
            if exists(mask):
                mask = F.pad(mask, (0, padding), value = False)
        
        # derive queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        
        # split out heads, and also divide sequence into blocks
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        
        if exists(mask):
            mask = repeat(mask, 'b n -> (b h) n', h = h)
        
        # scale
        q = q * self.scale
        
        # rotary pos emb
        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(pad_to_len, device = device))
            freqs = rearrange(freqs, 'n d -> () n d')
            q, k, v = map(lambda t: apply_rotary_emb(freqs, t), (q, k, v))
        
        # calculate number of levels until 2 x 2
        num_levels = int(log2(pad_to_len // bsz)) - 2
        assert num_levels >= 0, 'number of levels must be at least greater than 0'
        
        # coarsening
        qkvs = [(q, k, v, mask)]
        for level in range(num_levels):
            q, k, v = map(lambda t: rearrange(t, 'b (n r) d -> b n r d', r = 2), (q, k, v))
            if exists(mask):
                mask = repeat(mask, 'b (n r) -> b n r', r = 2)
            
            q = masked_aggregate(q, mask, dim = 2)
            k = masked_aggregate(k, mask, dim = 2)
            v = masked_aggregate(v, mask, dim = 2, average = False)
            if exists(mask):
                mask = torch.any(mask, dim = 2)
            
            coarsened_qkvs = (q, k, v, mask)
            qkvs.append(coarsened_qkvs)
        
        qkvs = [qkvs[0], *qkvs]  # duplicate the finest resolution an extra time, for the base diagonal
        # half-attention function

        def calculate_Y_and_A(q, k, v, mask = None):
            S = einsum('... i d, ... j d -> ... i j', q, k)

            if exists(mask):
                mask_value = -torch.finfo(S.dtype).max
                S = S.masked_fill(~mask, mask_value)

            S = S - torch.max(S, dim = -1, keepdim = True).values
            A = S.exp()

            y = einsum('... i j, ... j d -> ... i d', A, v)

            A = A.sum(dim = -1)

            y = rearrange(y, 'b ... n d -> b (... n) d')
            A = rearrange(A, 'b ... i -> b (... i)')
            return y, A

        to_blocks = lambda t: rearrange(t, 'b (n z) ... -> b n z ...', z = bsz)

        # calculate Ys, as in the paper

        Ys = []

        for ind, (q, k, v, mask) in enumerate(reversed(qkvs)):
            is_last = ind == (len(qkvs) - 1)

            q, k, v = map(to_blocks, (q, k, v))

            # generate the mask for S

            S_mask = None
            if exists(mask):
                mask = to_blocks(mask)
                q_mask = mask
                k_mask = cast_for_op(torch.int, flip_every_two)(mask) if not is_last else mask
                S_mask = rearrange(q_mask, '... n -> ... n ()') * rearrange(k_mask, '... n -> ... () n')

            # flip keys and values to capture the off-diagonals

            if not is_last:
                k, v = map(flip_every_two, (k, v))

            Y_level = calculate_Y_and_A(q, k, v, mask = S_mask)
            Ys.append(Y_level)

        # interpolate

        Y = 0
        A = 0

        for ind, (Y_level, A_level) in enumerate(Ys):
            is_last = ind == (len(Ys) - 1)

            if not is_last and torch.is_tensor(Y):
                Y = repeat(Y, 'b n d -> b (n r) d', r = 2)

            if not is_last and torch.is_tensor(A):
                A = repeat(A, 'b n -> b (n r)', r = 2)

            Y = Y_level + Y
            A = A_level + A

        out = Y / rearrange(A + eps, 'b n -> b n ()')

        # merge heads

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine out

        return self.to_out(out[:, :n])

class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)
        self.ih, self.iw = image_size
        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)
        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )
        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )
    def forward(self, x):
        _B, _C, _H, _W = x.shape
        x = x.view(128, 3072, 2)
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': BasicBlock, 'T': Transformer}
        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        print(self.s0)
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))
        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)
    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x
    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


def coatnet_0():
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_4():
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    net = coatnet_0()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_1()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_2()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_3()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_4()
    out = net(img)
    print(out.shape, count_parameters(net))
