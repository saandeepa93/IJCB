import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.transformer = Transformer(cfg.TRANSFORMER.DIM_OUT, cfg.TRANSFORMER.DEPTH, cfg.TRANSFORMER.HEADS\
            , cfg.TRANSFORMER.DIM_HEAD, cfg.TRANSFORMER.MLP_DIM, cfg.TRANSFORMER.DROPOUT)
        
        self.to_patch_embedding = nn.Sequential(
            # nn.LayerNorm(cfg.TRANSFORMER.DIM_IN),
            nn.Linear(cfg.TRANSFORMER.DIM_IN, cfg.TRANSFORMER.DIM_OUT),
            # nn.LayerNorm(cfg.TRANSFORMER.DIM_OUT),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.DATASET.NUM_FRAMES + 1, cfg.TRANSFORMER.DIM_OUT))
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.TRANSFORMER.DIM_OUT))
        self.pool = cfg.TRANSFORMER.POOL

    
    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        return x

class ViT_face(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        image_height = image_width = cfg.DATASET.IMG_SIZE
        patch_height = patch_width = cfg.TRANSFORMER.PATCH_SIZE

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = 3 * patch_height * patch_width
        dim = cfg.TRANSFORMER.DIM_OUT

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(cfg.TRANSFORMER.DROPOUT)
        
        self.transformer = Transformer(dim, cfg.TRANSFORMER.DEPTH, cfg.TRANSFORMER.HEADS\
            , cfg.TRANSFORMER.DIM_HEAD, cfg.TRANSFORMER.MLP_DIM, cfg.TRANSFORMER.DROPOUT)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, cfg.DATASET.N_CLASS)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) #if self.pool == 'mean' else x[:, 0]
        
        x = self.mlp_head(x)
        return x