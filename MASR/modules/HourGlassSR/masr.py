import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from . import block as B
import block as B
import pdb
import common


## Multi-path Adaptive Modulation(MAM) Layer
class MAMLayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(MAMLayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
        )
        # depthwise convolution
        self.csd = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        var = torch.var(x, dim=(2, 3), keepdim=True)
        var = F.normalize(var, p=2, dim=1)  # var normalization
        ca_out = self.conv_du(var)
        csd_out = self.csd(x)
        y = var + ca_out + csd_out
        y = self.act(y)
        return x * y


## Multi-path Adaptive Modulation Block (MAMB)
class MAMB(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction=16,
        bias=True, bn=False, act=nn.LeakyReLU(0.2, True), res_scale=1, conv_head=True):  # act=nn.ReLU(True)

        super(MAMB, self).__init__()
        modules_body = []
        if conv_head:
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            modules_body.append(act)
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias))
        modules_body.append(MAMLayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualBlock(nn.Module):
    """
    Residual BLock For SR without Norm Layer
    conv 1*1
    MAMB 3*3
    """

    def __init__(self, ch_in=64, ch_out=128, in_place=True):
        super(ResidualBlock, self).__init__()

        conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        conv2 = MAMB(n_feat=ch_out, kernel_size=3, conv_head=True)
        relu1 = nn.LeakyReLU(0.2, inplace=in_place)
        # relu2 = nn.LeakyReLU(0.2, inplace=in_place)

        # self.res = nn.Sequential(conv1, relu1, conv2, relu2, conv3)
        self.res = nn.Sequential(conv1, relu1, conv2)
        if ch_in != ch_out:
            self.identity = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0,
                                      bias=False)
        else:
            def identity(tensor):
                return tensor

            self.identity = identity

    def forward(self, x):
        res = self.res(x)
        # x = self.identity(x)
        # return torch.add(x, res)
        return res


class ResidualBlockFront(nn.Module):
    """
    conv 1*1
    MAMB 3*3
    MAMB 3*3
    """

    def __init__(self, ch_in=64, ch_out=128, in_place=True):
        super(ResidualBlockFront, self).__init__()

        conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        conv2 = MAMB(n_feat=ch_out, kernel_size=3, conv_head=True)
        conv3 = MAMB(n_feat=ch_out, kernel_size=3, conv_head=True)
        relu1 = nn.LeakyReLU(0.2, inplace=in_place)
        # relu2 = nn.LeakyReLU(0.2, inplace=in_place)

        # self.res = nn.Sequential(conv1, relu1, conv2, relu2, conv3)
        self.res = nn.Sequential(conv1, relu1, conv2, conv3)
        if ch_in != ch_out:
            self.identity = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0,
                                      bias=False)
        else:
            def identity(tensor):
                return tensor

            self.identity = identity

    def forward(self, x):
        res = self.res(x)
        # x = self.identity(x)
        # return torch.add(x, res)
        return res   # no long skip connection



class ResidualInceptionBlock(nn.Module):

    def __init__(self, ch_in=64, ch_out=128, n_resblocks=16, reduction=16, in_place=True):
        super(ResidualInceptionBlock, self).__init__()

        conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        act = nn.LeakyReLU(0.2, inplace=in_place)
        modules_body = [conv1, act]
        for i in range(n_resblocks):  #  8,12,16
            modules_body.append(MAMB(ch_out, kernel_size=3, reduction=reduction, bias=True, bn=False, conv_head=True))
        modules_body.append(nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True))
        self.res = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.res(x)
        x = x + res
        return x


class TopDownBlock(nn.Module):
    """
    Top to Down Block for HourGlass Block
    Consist of ConvNet Block and Pooling
    """

    def __init__(self, ch_in=64, ch_out=64, res_type='res'):
        super(TopDownBlock, self).__init__()
        if res_type == 'rrdb':
            self.res_block = B.RRDB(ch_in, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                    norm_type=None, act_type='leakyrelu', mode='CNA')
        else:
            self.res_block = ResidualBlockFront(ch_in=ch_in, ch_out=ch_out)
            # self.res_block = ResidualInceptionBlock(ch_in=ch_in, ch_out=ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.res_block(x)
        return self.pool(x), x


class BottomUpBlock(nn.Module):
    """
    Bottom Up Block for HourGlass Block
    Consist of ConvNet Block and Upsampling Block
    """

    def __init__(self, ch_in=64, ch_out=64, res_type='res'):
        super(BottomUpBlock, self).__init__()
        if res_type == 'rrdb':
            self.res_block = B.RRDB(ch_in, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                    norm_type=None, act_type='leakyrelu', mode='CNA')
        else:
            self.res_block = ResidualBlock(ch_in=ch_in, ch_out=ch_out)
            # self.res_block = ResidualInceptionBlock(ch_in=ch_in, ch_out=ch_out)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, res):
        x = self.upsample(x)
        return self.res_block(x + res)


class HourGlassBlock(nn.Module):
    """
    Hour Glass Block for SR Model
    """

    def __init__(self, res_type='res', n_mid=2, n_tail=2, n_resblocks=16):
        super(HourGlassBlock, self).__init__()
        self.n_tail = n_tail
        
        # l1, l2, l3, l4, l5 = 64, 96, 128, 192, 256
        # l1, l2, l3, l4, l5 = 64, 96, 128, 160, 192  # cdcxg2r
        l1, l2, l3, l4, l5 = 64, 96, 96, 128, 128  # best
        # l1, l2, l3, l4, l5 = 64, 80, 96, 112, 128
        # l1, l2, l3, l4, l5 = 64, 128, 128, 256, 256
        self.down1 = TopDownBlock(l1, l2, res_type=res_type)
        self.down2 = TopDownBlock(l2, l3, res_type=res_type)
        self.down3 = TopDownBlock(l3, l4, res_type=res_type)
        self.down4 = TopDownBlock(l4, l5, res_type=res_type)

        res_block = []
        for i in range(n_mid):
            res_block.append(ResidualBlock(l5, l5))
        self.mid_res = nn.Sequential(*res_block)

        self.up1 = BottomUpBlock(l5, l4, res_type=res_type)
        self.up2 = BottomUpBlock(l4, l3, res_type=res_type)
        self.up3 = BottomUpBlock(l3, l2, res_type=res_type)
        self.up4 = BottomUpBlock(l2, l1, res_type=res_type)

        if n_tail != 0:
            tail_block = []
            for i in range(n_tail):
                tail_block.append(ResidualInceptionBlock(64, 64, n_resblocks))
            self.tail = nn.Sequential(*tail_block)

    def forward(self, x):
        out, res1 = self.down1(x)
        out, res2 = self.down2(out)
        out, res3 = self.down3(out)
        out, res4 = self.down4(out)

        out = self.mid_res(out)

        out = self.up1(out, res4)
        out = self.up2(out, res3)
        out = self.up3(out, res2)
        out = self.up4(out, res1)
        out_inter = x + out

        if self.n_tail != 0:
            out = self.tail(out_inter)
        else:
            out = out_inter
        # Change to Residul Structure
        return out, out_inter


class CompMask(nn.Module):
    def __init__(self, upscale, in_ch, out_ch, kernel_size):
        super(CompMask, self).__init__()

        def make_upsample_block(upscale=upscale, in_ch=in_ch, kernel_size=kernel_size):
            n_upscale = 1 if upscale == 3 else int(math.log(upscale, 2))
            LR_conv = B.conv_block(in_ch, in_ch, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
            HR_conv0 = B.conv_block(in_ch, in_ch, kernel_size=3, norm_type=None, act_type='leakyrelu')
            if upscale == 1:
                return nn.Sequential(LR_conv, HR_conv0)
            elif upscale == 3:
                upsampler = B.upconv_blcok(in_ch, in_ch, 3, act_type='leakyrelu')
            else:
                upsampler = [B.upconv_blcok(in_ch, in_ch, act_type='leakyrelu') for _ in range(n_upscale)]
            # upsampler = common.Upsampler(common.default_conv, upscale, in_ch, act=False)  # 'relu'
            return nn.Sequential(LR_conv, *upsampler, HR_conv0)

        self.upsample = make_upsample_block(upscale=upscale, kernel_size=kernel_size)
        self.avg_atten = nn.AvgPool2d(3, stride=1, padding=1)
        self.max_atten = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv = nn.Sequential(nn.Conv2d(2*in_ch, out_ch, kernel_size=3, padding=1, stride=1), nn.Sigmoid())

        # self.conv = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1), nn.Sigmoid())  # 7
        # self.outconv = nn.Sequential(nn.Conv2d(in_ch, 3, kernel_size=3, padding=1, stride=1), nn.Sigmoid())

    def forward(self, x):
        x = self.upsample(x)
        y1 = self.avg_atten(x)
        y2 = self.max_atten(x)
        y = torch.cat((y1, y2), dim=1)
        y = self.conv(y)

        # y1, _ = torch.max(x, dim=1, keepdim=True)
        # y2 = torch.mean(x, dim=1, keepdim=True)
        # y = self.conv(torch.cat((y1, y2), dim=1))
        # y = self.outconv(x * y)

        # y = self.outconv(x)
        return y


class HourGlassNetMultiScaleInt(nn.Module):
    """
    Hour Glass SR Model, Use Mutil-Scale Label(HR_down_Xn) Supervision.
    """

    def __init__(self, in_nc=3, out_nc=3, upscale=4, nf=64, res_type='res', n_mid=2, n_tail=1, n_HG=6, n_resblocks=16,
                 act_type='leakyrelu', share_upsample=False):
        super(HourGlassNetMultiScaleInt, self).__init__()

        self.n_HG = n_HG
        self.n_resblocks = n_resblocks
        # modify
        self.in_nc = in_nc
        self.out_nc = out_nc
        ksize = 3

        self.conv_in = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)

        def make_upsample_block(upscale=4, in_ch=64, out_nc=out_nc, kernel_size=3):  # out_nc=3
            LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
            HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type='leakyrelu')
            HR_conv1 = B.conv_block(nf, out_nc, kernel_size=kernel_size, norm_type=None, act_type=None)

            n_upscale = 1 if upscale == 3 else int(math.log(upscale, 2))
            if upscale == 1:
                return nn.Sequential(LR_conv, HR_conv0, HR_conv1)
            elif upscale == 3:
                upsampler = B.upconv_blcok(nf, nf, 3, act_type=act_type)
            else:
                upsampler = [B.upconv_blcok(nf, nf, act_type=act_type) for _ in range(n_upscale)]
            # upsampler = common.Upsampler(common.default_conv, upscale, in_ch, act='relu')   # 'relu'/False
            return nn.Sequential(LR_conv, *upsampler, HR_conv0, HR_conv1)
        
        
        #Actually, the ksize can be any size for all scales
        self.flat_map = CompMask(upscale=upscale, kernel_size=ksize, in_ch=64, out_ch=3)
        self.edge_map = CompMask(upscale=upscale, kernel_size=ksize, in_ch=64, out_ch=3)
        self.corner_map = CompMask(upscale=upscale, kernel_size=ksize, in_ch=64, out_ch=3)
        
        self.upsample_flat = make_upsample_block(upscale=upscale)
        self.upsample_edge = make_upsample_block(upscale=upscale)
        self.upsample_corner = make_upsample_block(upscale=upscale)

        self.sub_mean = common.MeanShift(1.0, sign=-1)
        self.add_mean = common.MeanShift(1.0, sign=1)

        for i in range(n_HG):
            if i != n_HG - 1:
                HG_block = HourGlassBlock(res_type=res_type, n_mid=n_mid, n_tail=n_tail, n_resblocks=n_resblocks)
            else:
                HG_block = HourGlassBlock(res_type=res_type, n_mid=n_mid, n_tail=0, n_resblocks=n_resblocks)
            setattr(self, 'HG_%d' % i, HG_block)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.conv_in(x)
        SR_map = []
        result = []
        out = x

        # Multi-Scale supervise, [2, 2, 2] for 6, [2, 3, 3] for 8
        # super_block_idx = [1, self.n_HG // 2, self.n_HG - 1] # nhg>4
        super_block_idx = [0, 1, self.n_HG - 1]  # nhg=3/4

        for i in range(self.n_HG):
            out, out_inter = getattr(self, 'HG_%d' % i)(out)
            if i in super_block_idx:
                if i == self.n_HG - 1:
                    sr_feature = out.mul(0.2) + x  # 0.2
                else:
                    sr_feature = out_inter

                if super_block_idx.index(i) == 0:
                    srout_flat = self.upsample_flat(sr_feature)
                    srout_flat = self.add_mean(srout_flat)
                    flat_map = self.flat_map(sr_feature)
                    result.append(srout_flat)
                elif super_block_idx.index(i) == 1:
                    srout_edge = self.upsample_edge(sr_feature)
                    srout_edge = self.add_mean(srout_edge)
                    edge_map = self.edge_map(sr_feature)
                    result.append(srout_edge)
                elif super_block_idx.index(i) == 2:
                    srout_corner = self.upsample_corner(sr_feature)
                    srout_corner = self.add_mean(srout_corner)
                    corner_map = self.corner_map(sr_feature)
                    result.append(srout_corner)

                    srout = flat_map*srout_flat + edge_map*srout_edge + corner_map*srout_corner
                    result.append(srout)
                    SR_map.append(flat_map)
                    SR_map.append(edge_map)
                    SR_map.append(corner_map)

        return result, SR_map






