import torch
from torch import nn
import math


class ImgPatches(nn.Module):
    def __init__(self, input_channel=3, dim=1024, patch_size=8, num_patches_1d=32, num_patches_2d=32):
        super().__init__()
        self.num_patches_1d = num_patches_1d
        self.num_patches_2d = num_patches_2d
        self.patch_embed = nn.Conv2d(input_channel, dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches_1d * num_patches_2d, dim))

    def forward(self, img):
        patches = self.patch_embed(img)
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches + self.positional_embedding
        return patches


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1. / dim ** 0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)


class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, att_heads, dim, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, mlp_ratio, drop_rate) for heads in att_heads
        ])

    def forward(self, x, current_incremental_layer_index):
        for i in range(current_incremental_layer_index):
            x = self.Encoder_Blocks[i](x)
        return x


class CrossDomainAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, cross_domain_features):
        B, N, D = x.shape
        
        q = self.q(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(cross_domain_features).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(cross_domain_features).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.out(out)
        
        return out


class CrossDomainTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = CrossDomainAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout=dropout)
        
    def forward(self, x, cross_domain_features):
        x = x + self.cross_attn(self.norm1(x), cross_domain_features)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossDomainTransformerEncoder(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.1, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossDomainTransformerBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, cross_domain_features):
        for layer in self.layers:
            x = layer(x, cross_domain_features)
        return x


class ConvolutionBlockG(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_activation=True, use_norm=True, **kwargs):
        super(ConvolutionBlockG, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True) if use_activation else nn.Identity(),
        )

    def forward(self, x):
        return self.convolution(x)


class Generator(nn.Module):
    def __init__(self, img_channels=3, width=256, height=256, patch_sizes=[4,8,16,32], dim=1024,
                 mlp_ratio=4, drop_rate=0., att_heads=[2,4,8,16], use_cross_domain=True):
        super(Generator, self).__init__()
        self.height = height
        self.width = width
        self.img_channels = img_channels
        self.patch_sizes = patch_sizes
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = drop_rate
        self.use_cross_domain = use_cross_domain
        self.patches = nn.ModuleDict([])
        self.up_blocks = nn.ModuleList([])
        self.last_layers = nn.ModuleList([])
        last_up_op_dim = dim
        for patch_size in patch_sizes:
            if width % patch_size != 0 or height % patch_size != 0:
                raise ValueError('Image size must be divisible by patch size.')
            num_patches_1d = height // patch_size
            num_patches_2d = width // patch_size
            patch = ImgPatches(img_channels, dim, patch_size, num_patches_1d, num_patches_2d)
            up_block = ConvolutionBlockG(last_up_op_dim, dim // (patch_size//2), down=False, kernel_size=3, stride=2, padding=1,output_padding=1)
            last_block = nn.Conv2d(dim // (patch_size//2), img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
            last_up_op_dim = dim // (patch_size//2)
            self.patches[f"{patch_size}"] = patch
            self.up_blocks.append(up_block)
            self.last_layers.append(last_block)
        up_block = ConvolutionBlockG(last_up_op_dim, dim // patch_size, down=False, kernel_size=3, stride=2, padding=1,output_padding=1)
        last_block = nn.Conv2d(dim // patch_size, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        self.up_blocks.append(up_block)
        self.last_layers.append(last_block)
        self.TransformerEncoder = TransformerEncoder(att_heads=att_heads,dim=self.dim, mlp_ratio=self.mlp_ratio, drop_rate=self.dropout_rate)
        self.num_layers = len(att_heads)
        
        if self.use_cross_domain:
            self.cross_domain_encoder = CrossDomainTransformerEncoder(
                dim=dim, heads=8, mlp_ratio=mlp_ratio, dropout=drop_rate, num_layers=2
            )
            self.domain_fusion = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Dropout(drop_rate)
            )

    def forward(self, x, patch_size, cross_domain_features=None):
        x = self.patches[str(patch_size)](x)
        
        x = self.TransformerEncoder(x, self.num_layers)
        
        if self.use_cross_domain and cross_domain_features is not None:
            x_cross = self.cross_domain_encoder(x, cross_domain_features)
            
            x_fused = torch.cat([x, x_cross], dim=-1)
            x = self.domain_fusion(x_fused)
        
        x = x.permute(0, 2, 1).view(-1, self.dim, self.patches[str(patch_size)].num_patches_1d, self.patches[str(patch_size)].num_patches_2d)
        for idx in range(self.patch_sizes.index(patch_size)+2):
            x = self.up_blocks[idx](x)
        x = self.last_layers[self.patch_sizes.index(patch_size)+1](x)
        return torch.tanh(x)




class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 1024]):
        super(PatchDiscriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return self.model(x)





class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels),
            PatchDiscriminator(in_channels),
            PatchDiscriminator(in_channels)
        ])
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        results = []
        for D in self.discriminators:
            results.append(D(x))
            x = self.downsample(x)
        return results
