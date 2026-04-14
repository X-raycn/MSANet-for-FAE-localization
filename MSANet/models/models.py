import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.torch_dct import dct_2d, idct_2d


# ----------------------- 基础卷积 -----------------------
def conv3x3(in_c, out_c, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_c, out_c, 3, stride, padding, dilation, bias=False)


def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, bias=False)


# ----------------------- 基本块 -----------------------
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        if in_c != out_c or stride != 1:
            self.shortcut = nn.Sequential(
                conv1x1(in_c, out_c, stride),
                nn.BatchNorm2d(out_c)
            )
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        res = self.act(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        shortcut = getattr(self, 'shortcut', lambda t: t)(x)
        return self.act(res + shortcut)


# ----------------------- Channel Attention -----------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_c, ratio=8):
        super().__init__()
        mid_c = max(16, in_c // ratio)
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        self.conv1 = nn.Conv2d(in_c, mid_c, 1)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_c, in_c, 1)
        self.conv_w = nn.Conv2d(mid_c, in_c, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        idt = x
        x_h = self.pool_h_avg(x) + self.pool_h_max(x)
        x_w = self.pool_w_avg(x) + self.pool_w_max(x)
        x_w = x_w.permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        x_cat = self.act(self.bn1(self.conv1(x_cat)))
        x_cat = x_cat + x_cat.mean(2, keepdim=True)
        x_h, x_w = torch.split(x_cat, [x_h.size(2), x_w.size(2)], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out = self.sigmoid(self.conv_h(x_h) + self.conv_w(x_w))
        return idt * out + idt


# ----------------------- LKA + Transformer Spatial Attention -----------------------
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 7, padding=9, dilation=3, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv2(self.conv1(self.conv0(x)))
        return x * attn


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
        t2 = self.norm1(t)
        t = t + self.attn(t2, t2, t2, need_weights=False)[0]
        t2 = self.norm2(t)
        t = t + self.mlp(t2)
        return t.transpose(1, 2).reshape(B, C, H, W).contiguous()


class SpatialAttention(nn.Module):
    def __init__(self, in_c, pool_stride=8, heads=4, mlp_ratio=2.0, bottle=4):
        super().__init__()
        self.lka = LKA(in_c)
        c_b = max(32, in_c // bottle)
        self.pool = nn.AvgPool2d(pool_stride)
        self.upsample = nn.Upsample(scale_factor=pool_stride, mode='bilinear', align_corners=True)
        self.reduce = nn.Conv2d(in_c, c_b, 1)
        self.expand = nn.Conv2d(c_b, in_c, 1)
        self.trans = TransformerBlock(c_b, heads, mlp_ratio)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lka_out = self.lka(x)
        y = self.pool(lka_out)
        y = self.reduce(y)
        y = self.trans(y)
        y = self.expand(y)
        y = self.upsample(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return x * self.sigmoid(y) + x


# ============================================================
#  MSU：多尺度 + ConvNeXt
# ============================================================
class LayerNorm2d(nn.Module):
    """LayerNorm for [B,C,H,W]"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()   # [B,H,W,C]
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous() # [B,C,H,W]


class ConvNeXtBlock2d(nn.Module):
    """ConvNeXt style block (2022): DWConv(7x7)+LN+MLP(1x1)+LayerScale+Residual"""
    def __init__(self, dim, mlp_ratio=4.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)

        hidden = int(dim * mlp_ratio)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, 1)

        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim, 1, 1))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.gamma * x
        return shortcut + x


class MSU_MultiScaleConvNeXt(nn.Module):

    def __init__(self, dim, kernels=(3, 5, 7), dilations=(1, 2, 3),
                 reduction=16, post_blocks=2, mlp_ratio=4.0, use_lka=True):
        super().__init__()
        self.kernels = list(kernels)
        self.dilations = list(dilations)
        assert len(self.kernels) == len(self.dilations), "kernels 和 dilations 长度必须一致"
        k = len(self.kernels)

        # 多尺度分支
        branches = []
        for ki, di in zip(self.kernels, self.dilations):
            pad = (ki // 2) * di
            branches.append(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=ki, padding=pad, dilation=di, groups=dim, bias=False),  # DW
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),  # PW
                nn.BatchNorm2d(dim),
                nn.GELU()
            ))
        self.branches = nn.ModuleList(branches)

        # routing（给每个分支一个权重）
        mid = max(8, dim // reduction)
        self.route = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, k, 1, bias=True)
        )

        # 融合投影
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # modern mixing
        self.post = nn.Sequential(*[
            ConvNeXtBlock2d(dim, mlp_ratio=mlp_ratio, layer_scale_init=1e-6)
            for _ in range(post_blocks)
        ])

        self.lka = LKA(dim) if use_lka else nn.Identity()

        # gated residual
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        ys = [b(x) for b in self.branches]   # k * [B,C,H,W]
        ys = torch.stack(ys, dim=1)         # [B,k,C,H,W]

        w = self.route(x)                   # [B,k,1,1]
        w = torch.softmax(w, dim=1).unsqueeze(2)  # [B,k,1,1,1]

        y = (ys * w).sum(dim=1)             # [B,C,H,W]
        y = self.fuse(y)
        y = self.post(y)
        y = self.lka(y)

        g = self.gate(y)
        return x * (1 - g) + y * g


# ----------------------- AOTBlock（保留原样，方便回退/消融） -----------------------
class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super().__init__()
        self.rates = rates
        for i, r in enumerate(rates):
            self.add_module(
                f'block{i:02d}',
                nn.Sequential(
                    nn.ReflectionPad2d(r),
                    conv3x3(dim, dim // len(rates), padding=0, dilation=r),
                    nn.BatchNorm2d(dim // len(rates)),
                    nn.LeakyReLU(0.2, True)
                )
            )
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            conv3x3(dim, dim, padding=0),
            nn.BatchNorm2d(dim)
        )
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            conv3x3(dim, dim, padding=0),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        outs = [getattr(self, f'block{i:02d}')(x) for i in range(len(self.rates))]
        out = self.fuse(torch.cat(outs, 1))
        mask = torch.sigmoid(self.gate(x))
        return x * (1 - mask) + out * mask


# ----------------------- Down / Up Blocks -----------------------
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, block_num=1, down=True):
        super().__init__()
        if down:
            self.down = nn.Conv2d(in_c, in_c, 4, 2, 1)
        layers = [ResBlock(in_c, out_c)]
        for _ in range(block_num - 1):
            layers.append(ResBlock(out_c, out_c))
        self.layers = nn.Sequential(*layers)
        self.ca, self.sa = ChannelAttention(out_c), SpatialAttention(out_c)

    def forward(self, x):
        if hasattr(self, 'down'):
            x = self.down(x)
        x = self.sa(self.ca(self.layers(x)))
        return x


# ---- AttentionGate for skip connection ----
class AttentionGate(nn.Module):
    def __init__(self, in_skip, in_dec, inter):
        super().__init__()
        self.theta = nn.Conv2d(in_skip, inter, 1, bias=False)
        self.phi = nn.Conv2d(in_dec, inter, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.psi = nn.Conv2d(inter, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, skip, dec):
        g = self.act(self.theta(skip) + self.phi(dec))
        a = self.sig(self.psi(g))
        return skip * a


# ---- UpBlock with fusion ----
class UpBlockFuse(nn.Module):
    def __init__(self, in_dec, in_skip, out_c, up_mode='bilinear'):
        super().__init__()
        if up_mode == 'deconv':
            self.up = nn.ConvTranspose2d(in_dec, in_dec, 4, 2, 1)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True)
        inter = max(32, in_dec // 4)
        self.ag = AttentionGate(in_skip, in_dec, inter)
        self.fuse = ResBlock(in_dec + in_skip, out_c)
        self.ca, self.sa = ChannelAttention(out_c), SpatialAttention(out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        skip = self.ag(skip, x)
        x = torch.cat([x, skip], 1)
        x = self.sa(self.ca(self.fuse(x)))
        return x


# ---- Boundary Head（边缘特征 + 边缘图）----
class BoundaryHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 32):
        super().__init__()
        self.edge = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=1, padding=1,
            groups=in_channels, bias=False
        )
        sobel = torch.tensor([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]], dtype=torch.float32)
        with torch.no_grad():
            self.edge.weight.zero_()
            for c in range(in_channels):
                self.edge.weight[c, 0].copy_(sobel)

        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.out_edge = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        e = self.edge(x)
        e = torch.tanh(e)
        e = torch.abs(e)
        e = torch.clamp(e, 0, 5.0)

        feat = self.feat(e)          # [B, mid, H, W]
        edge = self.out_edge(feat)   # [B, 1, H, W]
        return edge, feat


# ----------------------- Classifier -----------------------
class Classifier(nn.Module):
    def __init__(self, in_c, n_class=2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_c, n_class)

    def forward(self, x):
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ----------------------- 主模型 Generator -----------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_channels = 32
        self.rates = [1, 2, 4, 8]          # 保留：方便消融/回退
        self.block_num = [4, 4, 4]         # 保留：方便消融/回退
        self.downsample_time = 2
        self.img_size = 256

        # 纯 RGB 输入 + 入口 CA
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, self.conv_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ChannelAttention(self.conv_channels)
        )

        # 编码器
        ch = self.conv_channels
        self.encoder = nn.ModuleList()
        self.enc_channels = []

        for i in range(self.downsample_time):
            use_lka = (i == self.downsample_time - 1)

            block = nn.Sequential(
                DownBlock(ch, ch * 2, 1, True),
                MSU_MultiScaleConvNeXt(
                    dim=ch * 2,
                    kernels=(3, 5, 7),
                    dilations=(1, 2, 3),
                    reduction=16,
                    post_blocks=2,
                    mlp_ratio=4.0,
                    use_lka=use_lka
                ),
            )
            self.encoder.append(block)
            ch *= 2
            self.enc_channels.append(ch)

        # bottleneck 额外再加一个全局 CA
        self.global_ca = ChannelAttention(ch)

        self.classifier = Classifier(ch, 2)

        # 解码器 (U-Net式)
        dec_in0, skip0, dec_out0 = ch, self.enc_channels[-2], ch // 2
        dec_in1, skip1, dec_out1 = dec_out0, self.conv_channels, dec_out0 // 2

        self.decoder = nn.ModuleList([
            UpBlockFuse(dec_in0, skip0, dec_out0),
            UpBlockFuse(dec_in1, skip1, dec_out1)
        ])

        # 边缘分支 + 边缘特征融合
        boundary_mid = 32
        self.boundary_head = BoundaryHead(dec_out1, mid_channels=boundary_mid)
        self.output_layer = nn.Conv2d(dec_out1 + boundary_mid, 1, 1)

    def forward(self, image):
        # 纯 RGB 流
        x0 = self.input_conv(image)

        enc_feats = []
        x = x0
        for layer in self.encoder:
            x = layer(x)
            enc_feats.append(x)

        # bottleneck 额外 CA
        enc_feats[-1] = self.global_ca(enc_feats[-1])

        # 分类分支使用最深特征
        bottleneck = enc_feats[-1]
        pred = self.classifier(bottleneck)

        # 解码器
        x = self.decoder[0](enc_feats[-1], enc_feats[-2])
        x = self.decoder[1](x, x0)

        # 边缘分支 + 边缘特征融合
        edge, edge_feat = self.boundary_head(x)
        x_cat = torch.cat([x, edge_feat], dim=1)
        mask = self.output_layer(x_cat)

        return mask, pred, edge
