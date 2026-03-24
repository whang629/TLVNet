import torch
import torch.nn as nn


class LCA(nn.Module):

    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        self.split_channels = [channels // groups] * groups

        self.conv_blocks = nn.ModuleList()
        for i in range(groups):
            if i == 0:
                self.conv_blocks.append(nn.Identity())
            else:
                self.conv_blocks.append(nn.Sequential(
                    nn.Conv2d(self.split_channels[i - 1], self.split_channels[i], 3, padding=1),
                    nn.LeakyReLU(0.2)
                ))

        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        splits = torch.split(x, self.split_channels, dim=1)
        features = [splits[0]]

        for i in range(1, self.groups):
            processed = self.conv_blocks[i](features[i - 1])
            features.append(processed + splits[i])

        fused = torch.cat(features, dim=1)
        return torch.sigmoid(self.fusion(fused)) * x


class GSA(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.conv = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x):
        avg = self.shared_mlp(self.avg_pool(x))
        max = self.shared_mlp(self.max_pool(x))
        channel_att = torch.sigmoid(self.conv(torch.cat([avg, max], dim=1)))
        return channel_att * x


class SRA(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.conv(x)
        return attn * x


class HAM(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.lca = LCA(channels)
        self.gsa = GSA(channels)
        self.sra = SRA(channels)

        self.fusion = nn.Conv2d(3 * channels, channels, 1)

    def forward(self, x):
        f1 = self.lca(x)
        f2 = self.gsa(f1)
        f3 = self.sra(f2)
        fused = self.fusion(torch.cat([f1, f2, f3], dim=1))
        return fused + x
