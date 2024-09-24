"""
Basic blocks for models
"""

import torch
from torch import nn
import torch.nn.functional as functional
from einops import rearrange
from utils import complex_cat


######################################################################################################################
#                                                    CONV blocks                                                     #
######################################################################################################################
# causal convolution
class causalConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=True):
        super(causalConv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=(padding[0], 0),
                              dilation=dilation, groups=groups, bias=bias)
        self.padding = padding[1]

    def forward(self, x):
        x = functional.pad(x, [self.padding, 0, 0, 0])

        out = self.conv(x)
        return out


# convolution block
class CONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CONV, self).__init__()
        self.conv = causalConv2d(in_ch, out_ch, kernel_size=(3, 2), stride=(2, 1), padding=(1, 1))
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ln(self.conv(x)))


# complex convolution block with complex operation
class CCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CCONV, self).__init__()
        self.conv_r = causalConv2d(in_ch // 2, out_ch // 2, kernel_size=(3, 2), stride=(2, 1), padding=(1, 1))
        self.ln_r = nn.GroupNorm(1, out_ch // 2, eps=1e-8)
        self.prelu_r = nn.PReLU()

        self.conv_i = causalConv2d(in_ch // 2, out_ch // 2, kernel_size=(3, 2), stride=(2, 1), padding=(1, 1))
        self.ln_i = nn.GroupNorm(1, out_ch // 2, eps=1e-8)
        self.prelu_i = nn.PReLU()

    def forward(self, x_r, x_i):
        r2r = self.conv_r(x_r)
        r2i = self.conv_i(x_r)
        i2i = self.conv_i(x_i)
        i2r = self.conv_r(x_i)

        real_out = r2r - i2i
        imag_out = i2r + r2i

        real_out = self.prelu_r(self.ln_r(real_out))
        imag_out = self.prelu_i(self.ln_i(imag_out))
        return real_out, imag_out


# convolution block for input layer
class INCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(INCONV, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ln(self.conv(x)))


# convolution block for input layer with complex operation
class CINCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CINCONV, self).__init__()
        self.conv_r = nn.Conv2d(in_ch // 2, out_ch // 2, kernel_size=1)
        self.ln_r = nn.GroupNorm(1, out_ch // 2, eps=1e-8)
        self.prelu_r = nn.PReLU()

        self.conv_i = nn.Conv2d(in_ch // 2, out_ch // 2, kernel_size=1)
        self.ln_i = nn.GroupNorm(1, out_ch // 2, eps=1e-8)
        self.prelu_i = nn.PReLU()

    def forward(self, x_r, x_i):
        r2r = self.conv_r(x_r)
        r2i = self.conv_i(x_r)
        i2i = self.conv_i(x_i)
        i2r = self.conv_r(x_i)

        x_r = r2r - i2i
        x_i = i2r + r2i

        x_r = self.prelu_r(self.ln_r(x_r))
        x_i = self.prelu_i(self.ln_r(x_i))
        return x_r, x_i


class COUTCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(COUTCONV, self).__init__()
        self.conv_r = nn.Conv2d(in_ch // 2, out_ch // 2, kernel_size=1)
        self.conv_i = nn.Conv2d(in_ch // 2, out_ch // 2, kernel_size=1)

    def forward(self, x_r, x_i):
        r2r = self.conv_r(x_r)
        r2i = self.conv_i(x_r)
        i2i = self.conv_i(x_i)
        i2r = self.conv_r(x_i)

        x_r = r2r - i2i
        x_i = i2r + r2i
        return x_r, x_i


######################################################################################################################
#                                         Down & Up sampling blocks                                                  #
######################################################################################################################
# 1x1 conv for down-sampling
class down_sampling(nn.Module):
    def __init__(self, in_ch):
        super(down_sampling, self).__init__()
        self.down_sampling = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 1), stride=(2, 1), padding=(0, 0))

    def forward(self, x):
        return self.down_sampling(x)


# 1x1 conv for down-sampling with complex operation
class complex_down_sampling(nn.Module):
    def __init__(self, in_ch):
        super(complex_down_sampling, self).__init__()
        self.down_sampling_r = nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=(1, 1), stride=(2, 1), padding=(0, 0))
        self.down_sampling_i = nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=(1, 1), stride=(2, 1), padding=(0, 0))

    def forward(self, x_r, x_i):
        r2r = self.down_sampling_r(x_r)
        r2i = self.down_sampling_i(x_r)
        i2i = self.down_sampling_i(x_i)
        i2r = self.down_sampling_r(x_i)

        x_r = r2r - i2i
        x_i = i2r + r2i

        return x_r, x_i


class BAND_complex_down_sampling(nn.Module):
    def __init__(self, in_ch):
        super(BAND_complex_down_sampling, self).__init__()
        self.down_sampling_1_1 = complex_down_sampling(in_ch)
        self.down_sampling_1_2 = complex_down_sampling(in_ch)

        self.down_sampling_2_1 = complex_down_sampling(in_ch)
        self.down_sampling_2_2 = complex_down_sampling(in_ch)

        self.down_sampling_3 = complex_down_sampling(in_ch)

        self.conv_r = nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1, stride=1)
        self.conv_i = nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1, stride=1)

    def forward(self, x_r, x_i):
        x_r1, x_r2, x_r3, x_r4 = torch.chunk(x_r, 4, dim=2)
        x_i1, x_i2, x_i3, x_i4 = torch.chunk(x_i, 4, dim=2)

        x_r1 = self.conv_r(x_r1)
        x_i1 = self.conv_i(x_i1)

        x_r2, x_i2 = self.down_sampling_3(x_r2, x_i2)

        x_r3, x_i3 = self.down_sampling_2_1(x_r3, x_i3)
        x_r3, x_i3 = self.down_sampling_2_1(x_r3, x_i3)

        x_r4, x_i4 = self.down_sampling_1_1(x_r4, x_i4)
        x_r4, x_i4 = self.down_sampling_1_2(x_r4, x_i4)

        x_r = torch.cat([x_r1, x_r2, x_r3, x_r4], dim=2)
        x_i = torch.cat([x_i1, x_i2, x_i3, x_i4], dim=2)
        return x_r, x_i


# sub-pixel convolution block
class SPCONV(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(SPCONV, self).__init__()
        self.conv = causalConv2d(in_ch, out_ch * scale_factor, kernel_size=(3, 2), padding=(1, 1))
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

        self.n = scale_factor

    def forward(self, x):
        x = self.conv(x)  # [B, C, F, T]

        r = rearrange(x, 'b (c n) f t -> b c (f n) t', n=2)

        out = self.ln(r)
        out = self.prelu(out)
        return out


# complex sub-pixel convolution block with complex operation
class CSPCONV(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(CSPCONV, self).__init__()
        self.conv_r = causalConv2d(in_ch // 2, out_ch // 2 * scale_factor, kernel_size=(3, 2), padding=(1, 1))
        self.ln_r = nn.GroupNorm(1, out_ch // 2, eps=1e-8)
        self.prelu_r = nn.PReLU()

        self.conv_i = causalConv2d(in_ch // 2, out_ch // 2 * scale_factor, kernel_size=(3, 2), padding=(1, 1))
        self.ln_i = nn.GroupNorm(1, out_ch // 2, eps=1e-8)
        self.prelu_i = nn.PReLU()

        self.n = scale_factor

    def forward(self, x_r, x_i):
        r2r = self.conv_r(x_r)
        r2i = self.conv_i(x_r)
        i2i = self.conv_i(x_i)
        i2r = self.conv_r(x_i)

        # for real component
        x_r = r2r - i2i

        r_r = rearrange(x_r, 'b (c n) f t -> b c (f n) t', n=2)

        out_r = self.ln_r(r_r)
        out_r = self.prelu_r(out_r)

        # for imag component
        x_i = i2r + r2i

        r_i = rearrange(x_i, 'b (c n) f t -> b c (f n) t', n=2)

        out_i = self.ln_i(r_i)
        out_i = self.prelu_i(out_i)

        return out_r, out_i



######################################################################################################################
#                                               Bottleneck blocks                                                    #
######################################################################################################################
class dilatedDenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_layers, inner=False):
        super(dilatedDenseBlock, self).__init__()

        self.input_layer = causalConv2d(in_ch, in_ch // 2, kernel_size=(3, 2), padding=(1, 1))  # channel half
        self.prelu1 = nn.PReLU()

        # dilated dense layer
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.caus_padd = ((2 ** i) // 2) * 2
            if i == 0: self.caus_padd = 1

            self.layers.append(nn.Sequential(
                # depth-wise separable conv
                causalConv2d(in_ch // 2 + i * in_ch // 2, in_ch // 2 + i * in_ch // 2, kernel_size=(3, 2),
                             padding=(2 ** i, self.caus_padd), dilation=2 ** i, groups=in_ch // 2 + i * in_ch // 2),
                # depth-wise
                nn.Conv2d(in_ch // 2 + i * in_ch // 2, in_ch // 2, kernel_size=1),  # pointwise
                nn.GroupNorm(1, in_ch // 2, eps=1e-8),
                nn.PReLU()
            ))

        self.out_layer = causalConv2d(in_ch // 2, out_ch, kernel_size=(3, 2), padding=(1, 1))  # channel revert
        if inner:
            self.prelu2 = nn.PReLU()
        else:
            self.prelu2 = nn.PReLU(out_ch)

    def forward(self, x):
        x = self.input_layer(x)  # C: in_ch//2
        x = self.prelu1(x)

        out = self.layers[0](x)

        pre_out = torch.cat([out, x], dim=1)
        for idx in range(len(self.layers) - 1):
            out = self.layers[idx + 1](pre_out)
            pre_out = torch.cat([out, pre_out], dim=1)

        out = self.out_layer(out)
        out = self.prelu2(out)

        return out


# complex dilated dense block with complex operation
class complexDilatedDenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_layers, inner=False):
        super(complexDilatedDenseBlock, self).__init__()

        self.input_layer_r = causalConv2d(in_ch // 2, in_ch // 4, kernel_size=(3, 2), padding=(1, 1))  # channel half
        self.input_layer_i = causalConv2d(in_ch // 2, in_ch // 4, kernel_size=(3, 2), padding=(1, 1))  # channel half
        self.prelu1_r = nn.PReLU()
        self.prelu1_i = nn.PReLU()

        # dilated dense layer
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.caus_padd = ((2 ** i) // 2) * 2
            if i == 0: self.caus_padd = 1

            self.layers.append(
                # depth-wise separable conv with complex operation
                CDSCONV(in_ch // 4 + i * in_ch // 4, in_ch // 4, kernel_size=(3, 2),
                        padding=(2 ** i, self.caus_padd), dilation=2 ** i)
            )

        self.out_layer_r = causalConv2d(in_ch // 4, out_ch // 2, kernel_size=(3, 2), padding=(1, 1))  # channel revert
        self.out_layer_i = causalConv2d(in_ch // 4, out_ch // 2, kernel_size=(3, 2), padding=(1, 1))  # channel revert
        if inner:
            self.prelu2_r = nn.PReLU()
            self.prelu2_i = nn.PReLU()
        else:
            self.prelu2_r = nn.PReLU(out_ch // 2)
            self.prelu2_i = nn.PReLU(out_ch // 2)

    def forward(self, x_r, x_i):
        r2r = self.input_layer_r(x_r)
        r2i = self.input_layer_i(x_r)
        i2i = self.input_layer_i(x_i)
        i2r = self.input_layer_r(x_i)

        x_r = r2r - i2i
        x_i = i2r + r2i

        x_r = self.prelu1_r(x_r)
        x_i = self.prelu1_i(x_i)

        out1_r, out1_i = self.layers[0](x_r, x_i)

        pre_out_r = torch.cat([out1_r, x_r], dim=1)
        pre_out_i = torch.cat([out1_i, x_i], dim=1)
        for idx in range(len(self.layers) - 1):
            out_r, out_i = self.layers[idx + 1](pre_out_r, pre_out_i)
            pre_out_r, pre_out_i = torch.cat([out_r, pre_out_r], dim=1), torch.cat([out_i, pre_out_i], dim=1)

        r2r = self.out_layer_r(out_r)
        r2i = self.out_layer_i(out_r)
        i2i = self.out_layer_i(out_i)
        i2r = self.out_layer_r(out_i)

        out_r = r2r - i2i
        out_i = i2r + r2i

        out_r = self.prelu2_r(out_r)
        out_i = self.prelu2_i(out_i)
        return out_r, out_i


# depth-wise separable conv with complex operation
class CDSCONV(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, dilation):
        super(CDSCONV, self).__init__()
        # depth-wise
        self.dwconv_r = causalConv2d(in_ch, in_ch, kernel_size=kernel_size,
                                     padding=padding, dilation=dilation, groups=in_ch)
        self.dwconv_i = causalConv2d(in_ch, in_ch, kernel_size=kernel_size,
                                     padding=padding, dilation=dilation, groups=in_ch)
        # separable
        self.sconv_r = nn.Conv2d(in_ch, out_ch, kernel_size=1)  # pointwise
        self.sconv_i = nn.Conv2d(in_ch, out_ch, kernel_size=1)  # pointwise

        self.ln_r = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.ln_i = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu_r = nn.PReLU()
        self.prelu_i = nn.PReLU()

    def forward(self, x_r, x_i):
        r2r = self.dwconv_r(x_r)
        r2r = self.sconv_r(r2r)
        r2i = self.dwconv_i(x_r)
        r2i = self.sconv_i(r2i)
        i2i = self.dwconv_i(x_i)
        i2i = self.sconv_i(i2i)
        i2r = self.dwconv_r(x_i)
        i2r = self.sconv_r(i2r)

        x_r = r2r - i2i
        x_i = i2r + r2i

        x_r = self.prelu_r(self.ln_r(x_r))
        x_i = self.prelu_i(self.ln_i(x_i))

        return x_r, x_i


######################################################################################################################
#                                                   Attention                                                        #
######################################################################################################################
# causal version of a time-frequency attention (TFA) module
# TFA paper reference: https://arxiv.org/pdf/2111.07518.pdf
class CTFA(nn.Module):
    def __init__(self, in_ch, out_ch=1, time_seq=32, kernel_size=17):
        super(CTFA, self).__init__()
        padding = kernel_size // 2

        # time attention
        self.time_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.time_conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.time_relu = nn.ReLU()
        self.time_conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.time_sigmoid = nn.Sigmoid()

        # frequency attention
        self.freq_avg_pool = nn.AvgPool1d(time_seq, stride=1)
        self.freq_conv1 = causalConv2d(in_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, kernel_size - 1))
        self.freq_relu = nn.ReLU()
        self.freq_conv2 = causalConv2d(out_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, kernel_size - 1))
        self.freq_sigmoid = nn.Sigmoid()

        # for real-time
        self.padd = time_seq - 1

    def forward(self, x):
        B, C, D, T = x.size()

        # time attention
        Z_T = x.permute(0, 1, 3, 2)
        Z_T = Z_T.reshape([B, C * T, D])
        TA = self.time_avg_pool(Z_T)  # [B, C*T, 1]
        TA = TA.reshape([B, C, T])
        TA = self.time_conv1(TA)
        TA = self.time_relu(TA)
        TA = self.time_conv2(TA)
        TA = self.time_sigmoid(TA)
        TA = TA.reshape([B, 1, 1, T]).expand(B, C, D, T)

        # frequency attention
        x_pad = functional.pad(x, [self.padd, 0, 0, 0])
        Z_F = x_pad.reshape([B, C * D, T + self.padd])
        FA = self.freq_avg_pool(Z_F)  # [B, C*F, T]
        FA = FA.reshape([B, C, D, T])
        FA = self.freq_conv1(FA)
        FA = self.freq_relu(FA)
        FA = self.freq_conv2(FA)
        FA = self.freq_sigmoid(FA)  # [B, C, D, T]

        # multiply
        TFA = FA * TA
        out = x * TFA

        return out


class CCTFA(nn.Module):
    def __init__(self, in_ch):
        super(CCTFA, self).__init__()

        self.ctfa_r = CTFA(in_ch // 2)
        self.ctfa_i = CTFA(in_ch // 2)

    def forward(self, x_r, x_i):
        out_r2r = self.ctfa_r(x_r)
        out_r2i = self.ctfa_i(x_r)
        out_i2r = self.ctfa_r(x_i)
        out_i2i = self.ctfa_i(x_i)
        return out_r2r - out_i2i, out_r2i + out_i2r