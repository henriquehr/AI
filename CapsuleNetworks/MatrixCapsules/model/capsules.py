import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PrimaryCaps(nn.Module):
    def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out


class ConvCaps(nn.Module):
    def __init__(self, B=32, C=32, K=3, P=4, stride=2, iters=3, coor_add=False, w_shared=False):
        super(ConvCaps, self).__init__()
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P * P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        self.eps = 1e-8
        self._lambda = 1e-03
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2 * math.pi))
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))
        self.weights = nn.Parameter(torch.randn(1, K * K * B, C, P, P))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize, inverse_temperature):
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu) ** 2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u.view(C, 1) + torch.log(sigma_sq.sqrt())) * r_sum

        #a_out = self.sigmoid(self._lambda * (self.beta_a - cost_h.sum(dim=2)))
        #self.beta_a.data = F.normalize(self.beta_a, p=2, dim=0)
        #b_a = F.normalize(inverse_temperature * (self.beta_a - cost_h.sum(dim=2)), p=2, dim=0)
        #b_a = F.normalize(self.beta_a, p=2, dim=0)
        #self.beta_a.data = b_a
        #a_out = self.sigmoid(b_a)
        a_out = self.sigmoid(inverse_temperature * (self.beta_a - cost_h.sum(dim=2)))
        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        ln_p_j_h = -1.0 * (v - mu) ** 2 / (2 * sigma_sq) - torch.log(sigma_sq.sqrt()) - 0.5 * self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        final_lambda = 0.01
        r = torch.cuda.FloatTensor(b, B, C).fill_(1.0 / C)
        for iter_ in range(self.iters):
            inverse_temperature= final_lambda * (1 - pow(0.95, iter_ + 1.0))
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize, inverse_temperature)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out

    def add_patches(self, x, B, K, psize, stride):
        b, h, w, c = x.shape
        assert h == w
        assert c == B * (psize + 1)
        oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) for h_idx in range(0, h - K + 1, stride)] for k_idx in range(0, K)]
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        b, B, psize = x.shape
        assert psize == P * P

        x = x.view(b, B, 1, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)
        v = v.view(b, B, C, P * P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = 1.0 * torch.arange(h) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.0)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.0)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h * w * B, C, psize)
        return v

    def forward(self, x):
        b, h, w, c = x.shape
        if not self.w_shared:
            x, oh, ow = self.add_patches(x, self.B, self.K, self.psize, self.stride)

            p_in = x[:, :, :, :, :, :self.B * self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B * self.psize:].contiguous()
            p_in = p_in.view(b * oh * ow, self.K * self.K * self.B, self.psize)
            a_in = a_in.view(b * oh * ow, self.K * self.K * self.B, 1)
            v = self.transform_view(p_in, self.weights, self.C, self.P)

            p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out = p_out.view(b, oh, ow, self.C * self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            out = torch.cat([p_out, a_out], dim=3)
        else:
            assert c == self.B * (self.psize + 1)
            assert 1 == self.K
            assert 1 == self.stride
            p_in = x[:, :, :, :self.B * self.psize].contiguous()
            p_in = p_in.view(b, h * w * self.B, self.psize)
            a_in = x[:, :, :, self.B * self.psize:].contiguous()
            a_in = a_in.view(b, h * w * self.B, 1)

            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)

            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            _, out = self.caps_em_routing(v, a_in, self.C, self.eps)

        return out


class CapsNet(nn.Module):
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=A, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=A)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters) 
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters, coor_add=True, w_shared=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x = self.class_caps(x)
        return x


def capsules(**kwargs):
    model = CapsNet(**kwargs)
    return model
