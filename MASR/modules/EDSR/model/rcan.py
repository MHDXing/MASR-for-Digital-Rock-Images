'''RCAN'''
from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res = x + res
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = x + res
        return res


class CompMask(nn.Module):
    def __init__(self, conv, scale, n_feats, out_channels, kernel_size):
        super(CompMask, self).__init__()
        self.upsample = common.Upsampler(conv, scale, n_feats, act=True)
        self.conv = nn.Sequential(conv(2*n_feats, out_channels, kernel_size), nn.Sigmoid())
        self.avg_atten = nn.AvgPool2d(5, stride=1, padding=2)
        self.max_atten = nn.MaxPool2d(5, stride=1, padding=2)
    
    def forward(self, x):
        x = self.upsample(x)
        x1 = self.avg_atten(x)
        x2 = self.max_atten(x)
        y = torch.cat((x1, x2), dim=1)
        y = self.conv(y)
        return y


class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # define head module
        modules_head = [conv(args.in_ch, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        def modules_tail(conv=conv, scale=scale, n_feats=n_feats, out_channels=args.out_ch, kernel_size=3):
            upsample = [
                common.Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, out_channels, kernel_size)]
            return nn.Sequential(*upsample)


        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = modules_tail()

        self.flat_mask = CompMask(conv=conv, scale=scale, n_feats=n_feats, out_channels=1, kernel_size=5)
        self.edge_mask = CompMask(conv=conv, scale=scale, n_feats=n_feats, out_channels=1, kernel_size=5)
        self.corner_mask = CompMask(conv=conv, scale=scale, n_feats=n_feats, out_channels=1, kernel_size=5)

        self.sub_mean = common.MeanShift(args.rgb_range, sign=-1)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)

        srout = self.tail(res + x)
        srout = self.add_mean(srout)

        result = [srout, srout, srout, srout]
        SR_map = result[:3]

        return result, SR_map 