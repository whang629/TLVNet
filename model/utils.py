import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


class ResBlock(nn.Module):

    def __init__(self, num_filter):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_filter, num_filter, 3, padding=0),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_filter, num_filter, 3, padding=0),
        )
        self.se = SEBlock(num_filter)
        self.gate = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        x = self.body(x)
        x = self.se(x)
        gate = self.gate(identity)
        return identity + x * gate


class ConvBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_out, 3, padding=0),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out, ch_out, 3, padding=0),
            nn.GELU(),
            SEBlock(ch_out)
        )

    def forward(self, x):
        return self.conv(x)


class SEBlock(nn.Module):

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.GELU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max = self.max_pool(x).view(b, c)
        weight = self.fc(avg + max).view(b, c, 1, 1)
        return x * weight.expand_as(x)


class Up(nn.Module):

    def __init__(self):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, 3, padding=1),
            SEBlock(64)
        )

    def forward(self, x):
        return self.up(x)


class TLE(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.u_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 2 * self.latent_dim, kernel_size=1)
        )

        self.s_encoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 2 * self.latent_dim, kernel_size=1)
        )

        self.v_encoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 2 * self.latent_dim, kernel_size=1)
        )

    def forward(self, x):
        u = self.u_encoder(x).squeeze(-1).squeeze(-1)
        s = self.s_encoder(x).squeeze(-1).squeeze(-1)
        v = self.v_encoder(x).squeeze(-1).squeeze(-1)

        u_mu, u_logvar = u.chunk(2, dim=1)
        s_mu, s_logvar = s.chunk(2, dim=1)
        v_mu, v_logvar = v.chunk(2, dim=1)

        u_logvar = torch.clamp(u_logvar, min=-10, max=10)
        s_logvar = torch.clamp(s_logvar, min=-10, max=10)
        v_logvar = torch.clamp(v_logvar, min=-10, max=10)

        u_std = torch.exp(0.5 * u_logvar)
        s_std = torch.exp(0.5 * s_logvar)
        v_std = torch.exp(0.5 * v_logvar)

        u_dist = Independent(Normal(u_mu, u_std), 1)
        s_dist = Independent(Normal(s_mu, s_std), 1)
        v_dist = Independent(Normal(v_mu, v_std), 1)

        return u_dist, s_dist, v_dist, u_mu, s_mu, v_mu, u_std, s_std, v_std
