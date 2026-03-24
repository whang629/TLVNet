import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from torch.distributions import kl
from model.utils import ResBlock, ConvBlock, Up, TLE
from model.HAM import HAM


class Encoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Sequential(ConvBlock(ch_in=ch, ch_out=64), HAM(64))
        self.conv2 = nn.Sequential(ConvBlock(64, 64), HAM(64))
        self.conv3 = nn.Sequential(ConvBlock(64, 64), HAM(64))
        self.conv4 = nn.Sequential(ResBlock(64), HAM(64))
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x2 = self.conv2(x2)
        x3 = self.pool2(x2)
        x3 = self.conv3(x3)
        x4 = self.pool3(x3)
        x4 = self.conv4(x4)
        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.pr_encoder = Encoder(3)
        self.po_encoder = Encoder(6)
        self.pr_conv = ResBlock(64)
        self.po_conv = ResBlock(64)
        self.pr_Up3 = Up()
        self.pr_UpConv3 = nn.Sequential(ConvBlock(128, 64), HAM(64))
        self.pr_Up2 = Up()
        self.pr_UpConv2 = nn.Sequential(ConvBlock(128, 64), HAM(64))
        self.pr_Up1 = Up()
        self.pr_UpConv1 = nn.Sequential(ConvBlock(128, 64), HAM(64))
        self.po_Up3 = Up()
        self.po_UpConv3 = ConvBlock(128, 64)
        self.po_Up2 = Up()
        self.po_UpConv2 = ConvBlock(128, 64)
        self.po_Up1 = Up()
        self.out_conv = nn.Sequential(
            HAM(64),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 3, kernel_size=1, padding=0)
        )
        z_dim = 20
        self.compute_z_pr = TLE(z_dim)
        self.compute_z_po = TLE(z_dim)
        self.conv_u = nn.Conv2d(z_dim, 128, 1)
        self.conv_s = nn.Conv2d(z_dim, 128, 1)
        self.conv_v = nn.Conv2d(z_dim, 128, 1)
        self.insnorm = nn.InstanceNorm2d(128)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Input, Target, training=True):
        pr_x1, pr_x2, pr_x3, pr_x4 = self.pr_encoder(Input)
        if training:
            po_x1, po_x2, po_x3, po_x4 = self.po_encoder(torch.cat((Input, Target), 1))
            pr_x4 = self.pr_conv(pr_x4)
            po_x4 = self.po_conv(po_x4)
            pr_d3 = self.pr_Up3(pr_x4)
            po_d3 = self.po_Up3(po_x4)
            pr_d3 = torch.cat((pr_x3, pr_d3), 1)
            po_d3 = torch.cat((po_x3, po_d3), 1)
            pr_d3 = self.pr_UpConv3(pr_d3)
            po_d3 = self.po_UpConv3(po_d3)
            pr_d2 = self.pr_Up2(pr_d3)
            po_d2 = self.po_Up2(po_d3)
            pr_d2 = torch.cat((pr_x2, pr_d2), 1)
            po_d2 = torch.cat((po_x2, po_d2), 1)
            pr_d2 = self.pr_UpConv2(pr_d2)
            po_d2 = self.po_UpConv2(po_d2)
            pr_d1 = self.pr_Up1(pr_d2)
            po_d1 = self.po_Up1(po_d2)
            pr_d1 = torch.cat((pr_x1, pr_d1), 1)
            po_d1 = torch.cat((po_x1, po_d1), 1)
            pr_u_dist, pr_s_dist, pr_v_dist, *_ = self.compute_z_pr(pr_d1)
            po_u_dist, po_s_dist, po_v_dist, *_ = self.compute_z_po(po_d1)
            po_u = self.conv_u(po_u_dist.rsample()[..., None, None])
            po_s = self.conv_s(po_s_dist.rsample()[..., None, None])
            po_v = self.conv_v(po_v_dist.rsample()[..., None, None])
            pr_d1 = self.insnorm(pr_d1) * torch.abs(po_s) * torch.abs(po_v) + po_u
            pr_d1 = self.pr_UpConv1(pr_d1)
            out = self.out_conv(pr_d1)
            return out, pr_u_dist, pr_s_dist, pr_v_dist, po_u_dist, po_s_dist, po_v_dist
        else:
            pr_x4 = self.pr_conv(pr_x4)
            pr_d3 = self.pr_Up3(pr_x4)
            pr_d3 = torch.cat((pr_x3, pr_d3), 1)
            pr_d3 = self.pr_UpConv3(pr_d3)
            pr_d2 = self.pr_Up2(pr_d3)
            pr_d2 = torch.cat((pr_x2, pr_d2), 1)
            pr_d2 = self.pr_UpConv2(pr_d2)
            pr_d1 = self.pr_Up1(pr_d2)
            pr_d1 = torch.cat((pr_x1, pr_d1), 1)
            pr_u_dist, pr_s_dist, pr_v_dist, *_ = self.compute_z_pr(pr_d1)
            pr_u = self.conv_u(pr_u_dist.rsample()[..., None, None])
            pr_s = self.conv_s(pr_s_dist.rsample()[..., None, None])
            pr_v = self.conv_v(pr_v_dist.rsample()[..., None, None])
            pr_d1 = self.insnorm(pr_d1) * torch.abs(pr_s) * torch.abs(pr_v) + pr_u
            pr_d1 = self.pr_UpConv1(pr_d1)
            out = self.out_conv(pr_d1)
            return out


class GFRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        self.conv1_o = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1_o = nn.BatchNorm2d(out_channels)
        self.conv2_o = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2_o = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.shortcut(x)
        x = F.relu(self.bn1_o(self.conv1_o(x)))
        x = F.relu(self.bn2_o(self.conv2_o(x)))
        x1 = self.conv1(x)
        x_mid = x1 + identity
        x1_bn = F.relu(self.bn1(x_mid))
        x2 = self.conv2(x1_bn)
        x2_bn = F.relu(self.bn2(x2))
        x3 = torch.cat([x, x2_bn], dim=1)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = torch.sigmoid(self.bn4(self.conv4(x3)))
        return x_mid * (1 - x4)


class MGR(nn.Module):
    def __init__(self, in_channels, output_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            *[GFRBlock(64, 64) for _ in range(5)],
            nn.Conv2d(64, output_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class PerceptionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            vgg = vgg16(pretrained=True)
            self.loss_net = nn.Sequential(*vgg.features[:31]).requires_grad_(False)
        except Exception as e:
            print(f"Failed to load pretrained VGG16: {e}")
            vgg = vgg16(pretrained=False)
            self.loss_net = nn.Sequential(*vgg.features[:31]).requires_grad_(False)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(self.loss_net(pred), self.loss_net(target))


class tlvnet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.device = torch.device(opt.device)

        self.decoder = Decoder(self.device).to(self.device)
        self.DB = MGR(in_channels=3, output_channels=3).to(self.device)

        self.mse = nn.MSELoss().to(self.device)
        self.vgg_loss = PerceptionLoss().to(self.device)
        self.beta = 0.0
        self.beta_max = 1.0
        self.beta_steps = 10000
        self.step = 0
        self.ortho_weight = 0.01
        self.edge_weight = 0.15
        self.to(self.device)

    def forward(self, Input, label, training=True):
        Input = Input.to(self.device)
        label = label.to(self.device)
        if training:
            out, *dists = self.decoder(Input, label, True)
            out = out - self.DB(out)
            self.out, (self.pr_u, self.pr_s, self.pr_v, self.po_u, self.po_s, self.po_v) = out, dists
        else:
            self.out = self.decoder(Input, label, False)
        return self.out

    def sample(self, input, label=None, testing=False):
        input = input.to(self.device)
        if testing:
            out = self.decoder(input, input, False)
        else:
            if label is None:
                raise ValueError("Label required for training mode sampling.")
            label = label.to(self.device)
            out, *_ = self.decoder(input, label, True)
        out = out - self.DB(out)
        return out

    def kl_divergence(self, analytic=True):
        beta = min(self.beta_max, self.beta + (self.beta_max / self.beta_steps) * self.step)
        self.step += 1
        kl_u = beta * torch.mean(kl.kl_divergence(self.po_u, self.pr_u))
        kl_s = beta * torch.mean(kl.kl_divergence(self.po_s, self.pr_s))
        kl_v = beta * torch.mean(kl.kl_divergence(self.po_v, self.pr_v))
        means = torch.cat([self.po_u.mean, self.po_s.mean, self.po_v.mean], dim=1)
        ortho_loss = torch.mean(torch.abs(torch.cov(means)))
        return kl_u + kl_s + kl_v + self.ortho_weight * ortho_loss

    def _sobel_edges(self, x):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device)
        sobel_y = sobel_x.t()
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        edge_x = F.conv2d(x, sobel_x, padding=1, groups=3)
        edge_y = F.conv2d(x, sobel_y, padding=1, groups=3)
        return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-10)

    def elbo(self, target, analytic_kl=True):
        target = target.to(self.device)
        pred_edges = self._sobel_edges(self.out)
        target_edges = self._sobel_edges(target)
        edge_loss = F.l1_loss(pred_edges, target_edges) * self.edge_weight
        return (
            self.mse(self.out, target) +
            self.vgg_loss(self.out, target) +
            self.kl_divergence(analytic_kl) +
            edge_loss
        )
