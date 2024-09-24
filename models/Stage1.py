import torch

from utils import ConvSTFT, ConviSTFT, power_compress, power_uncompress
from models.baseBlocks import *
from s3prl.nn import S3PRLUpstream
from einops import rearrange


class EncoderStage(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, layers=4, bottleneck_layers=6):
        super(EncoderStage, self).__init__()
        self.stage_num = layers

        # input layer
        self.inconv = CINCONV(in_ch, out_ch)
        # inner encoders
        self.encoders = nn.ModuleList(
            [CCONV(out_ch, mid_ch) if i == 0 else CCONV(mid_ch, mid_ch) for i in range(self.stage_num)])
        # inner bottleneck
        self.bottleneck = complexDilatedDenseBlock(mid_ch, mid_ch, bottleneck_layers, inner=True)
        # inner decoders
        self.decoders = nn.ModuleList(
            [CSPCONV(mid_ch * 2, out_ch) if i == self.stage_num - 1 else CSPCONV(mid_ch * 2, mid_ch) for i in
             range(self.stage_num)])
        # attention module
        self.att = CCTFA(out_ch)

        # down-sampling block
        self.downsampling = complex_down_sampling(out_ch)

    def forward(self, xr_in, xi_in):
        xr_in, xi_in = self.inconv(xr_in, xi_in)

        out_r, out_i = xr_in, xi_in
        encoder_outs_r, encoder_outs_i = [], []
        for idx, layers in enumerate(self.encoders):
            out_r, out_i = layers(out_r, out_i)
            encoder_outs_r.append(out_r), encoder_outs_i.append(out_i)

        out_r, out_i = self.bottleneck(out_r, out_i)

        for idx, layers in enumerate(self.decoders):
            out_r, out_i = layers(torch.cat([out_r, encoder_outs_r[-idx - 1]], dim=1),
                                  torch.cat([out_i, encoder_outs_i[-idx - 1]], dim=1))

        out_r, out_i = self.att(out_r, out_i)
        out_r = out_r + xr_in
        out_i = out_i + xi_in

        out_r, out_i = self.downsampling(out_r, out_i)
        return out_r, out_i


class DecoderStage(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, layers=4, bottleneck_layers=6):
        super(DecoderStage, self).__init__()
        self.stage_num = layers

        # up-sampling block
        self.upsampling = CSPCONV(in_ch * 2, in_ch * 2)

        # input layer
        self.inconv = CINCONV(in_ch * 2, out_ch)
        # inner encoders
        self.encoders = nn.ModuleList(
            [CCONV(out_ch, mid_ch) if i == 0 else CCONV(mid_ch, mid_ch) for i in range(self.stage_num)])
        # inner bottleneck
        self.bottleneck = complexDilatedDenseBlock(mid_ch, mid_ch, bottleneck_layers, inner=True)
        # inner decoders
        self.decoders = nn.ModuleList(
            [CSPCONV(mid_ch * 2, out_ch) if i == self.stage_num - 1 else CSPCONV(mid_ch * 2, mid_ch) for i in
             range(self.stage_num)])

        self.att = CCTFA(out_ch)

    def forward(self, xr_in, xi_in):
        xr_in, xi_in = self.upsampling(xr_in, xi_in)

        xr_in, xi_in = self.inconv(xr_in, xi_in)

        out_r, out_i = xr_in, xi_in
        encoder_outs_r, encoder_outs_i = [], []
        for idx, layers in enumerate(self.encoders):
            out_r, out_i = layers(out_r, out_i)
            encoder_outs_r.append(out_r), encoder_outs_i.append(out_i)

        out_r, out_i = self.bottleneck(out_r, out_i)

        for idx, layers in enumerate(self.decoders):
            out_r, out_i = layers(torch.cat([out_r, encoder_outs_r[-idx - 1]], dim=1),
                                  torch.cat([out_i, encoder_outs_i[-idx - 1]], dim=1))

        out_r, out_i = self.att(out_r, out_i)

        return out_r + xr_in, out_i + xi_in


class Stage1(nn.Module):
    def __init__(self, in_ch=2, mid_ch=64, out_ch=128,
                 WIN_LEN=512, HOP_LEN=256, FFT_LEN=512):
        super(Stage1, self).__init__()
        self.fft_half = FFT_LEN // 2 + 1

        # Input layer
        self.input_layer = CINCONV(in_ch, out_ch)

        # Encoder stages
        self.en1 = EncoderStage(out_ch, mid_ch, out_ch, layers=6)
        self.en2 = EncoderStage(out_ch, mid_ch, out_ch, layers=5)
        self.en3 = EncoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.en4 = EncoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.en5 = EncoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.en6 = EncoderStage(out_ch, mid_ch, out_ch, layers=3)

        # Bottleneck block
        self.bottleneck = complexDilatedDenseBlock(out_ch, out_ch, 6)

        # Decoder stages
        self.de1m = DecoderStage(out_ch, mid_ch, out_ch, layers=3)
        self.de2m = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de3m = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de4m = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de5m = DecoderStage(out_ch, mid_ch, out_ch, layers=5)
        self.de6m = DecoderStage(out_ch, mid_ch, out_ch, layers=6)

        self.de1s = DecoderStage(out_ch, mid_ch, out_ch, layers=3)
        self.de2s = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de3s = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de4s = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de5s = DecoderStage(out_ch, mid_ch, out_ch, layers=5)
        self.de6s = DecoderStage(out_ch, mid_ch, out_ch, layers=6)

        # output layer
        self.output_layer_m = COUTCONV(out_ch, in_ch)
        self.output_layer_s = COUTCONV(out_ch, in_ch)

        # for feature extract
        self.cstft = ConvSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='complex')
        self.cistft = ConviSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='complex')

        ############### SRL model
        self.conv1x1r = nn.Conv1d(1024 + 128, 128, kernel_size=1)
        self.norm_r = nn.GroupNorm(1, 128, eps=1e-8)
        self.conv1x1i = nn.Conv1d(1024 + 128, 128, kernel_size=1)
        self.norm_i = nn.GroupNorm(1, 128, eps=1e-8)

    def forward(self, real, imag, srl_features):
        real = real.unsqueeze(1)[:, :, 1:]
        imag = imag.unsqueeze(1)[:, :, 1:]

        # input layer
        hx_r, hx_i = self.input_layer(real, imag)

        # encoder stages
        hx1r, hx1i = self.en1(hx_r, hx_i)
        hx2r, hx2i = self.en2(hx1r, hx1i)
        hx3r, hx3i = self.en3(hx2r, hx2i)
        hx4r, hx4i = self.en4(hx3r, hx3i)
        hx5r, hx5i = self.en5(hx4r, hx4i)
        hx6r, hx6i = self.en6(hx5r, hx5i)

        srl_features = srl_features.permute(0, 2, 1)
        srl_features = functional.interpolate(srl_features, hx6r.size(-1), mode='linear')

        out_r = rearrange(hx6r, 'b c f t -> b (c f) t')
        out_r = self.norm_r(self.conv1x1r(torch.cat([out_r, srl_features], dim=1)))
        out_i = rearrange(hx6i, 'b c f t -> b (c f) t')
        out_i = self.norm_i(self.conv1x1i(torch.cat([out_i, srl_features], dim=1)))
        out_r = rearrange(out_r, 'b (c f) t -> b c f t', f=4)
        out_i = rearrange(out_i, 'b (c f) t -> b c f t', f=4)

        # dilated dense block
        out_r, out_i = self.bottleneck(out_r, out_i)

        # decoder stages - masking
        out_r_m, out_i_m = self.de1m(torch.cat([out_r, hx6r], dim=1), torch.cat([out_i, hx6i], dim=1))
        out_r_m, out_i_m = self.de2m(torch.cat([out_r_m, hx5r], dim=1), torch.cat([out_i_m, hx5i], dim=1))
        out_r_m, out_i_m = self.de3m(torch.cat([out_r_m, hx4r], dim=1), torch.cat([out_i_m, hx4i], dim=1))
        out_r_m, out_i_m = self.de4m(torch.cat([out_r_m, hx3r], dim=1), torch.cat([out_i_m, hx3i], dim=1))
        out_r_m, out_i_m = self.de5m(torch.cat([out_r_m, hx2r], dim=1), torch.cat([out_i_m, hx2i], dim=1))
        out_r_m, out_i_m = self.de6m(torch.cat([out_r_m, hx1r], dim=1), torch.cat([out_i_m, hx1i], dim=1))

        # decoder stages - mapping
        out_r_s, out_i_s = self.de1s(torch.cat([out_r, hx6r], dim=1), torch.cat([out_i, hx6i], dim=1))
        out_r_s, out_i_s = self.de2s(torch.cat([out_r_s, hx5r], dim=1), torch.cat([out_i_s, hx5i], dim=1))
        out_r_s, out_i_s = self.de3s(torch.cat([out_r_s, hx4r], dim=1), torch.cat([out_i_s, hx4i], dim=1))
        out_r_s, out_i_s = self.de4s(torch.cat([out_r_s, hx3r], dim=1), torch.cat([out_i_s, hx3i], dim=1))
        out_r_s, out_i_s = self.de5s(torch.cat([out_r_s, hx2r], dim=1), torch.cat([out_i_s, hx2i], dim=1))
        out_r_s, out_i_s = self.de6s(torch.cat([out_r_s, hx1r], dim=1), torch.cat([out_i_s, hx1i], dim=1))

        # output layer
        real_out_m, imag_out_m = self.output_layer_m(out_r_m, out_i_m)
        real_out_s, imag_out_s = self.output_layer_s(out_r_s, out_i_s)

        mask_mags = torch.sqrt(real_out_m ** 2 + imag_out_m ** 2)
        phase_real = real_out_m / (mask_mags + 1e-8)
        phase_imag = imag_out_m / (mask_mags + 1e-8)
        mask_phase = torch.atan2(phase_imag, phase_real)

        mask_mags = torch.tanh(mask_mags)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase = torch.atan2(imag, real)

        mag_out = mag * mask_mags
        phase_out = phase + mask_phase

        real_out_m = mag_out * torch.cos(phase_out)
        imag_out_m = mag_out * torch.sin(phase_out)

        real_out_m = functional.pad(real_out_m, [0, 0, 1, 0]).squeeze(1)
        imag_out_m = functional.pad(imag_out_m, [0, 0, 1, 0]).squeeze(1)

        out_specs_m = power_uncompress(real_out_m, imag_out_m)
        outputs_m = self.cistft(out_specs_m)

        real_out_s = functional.pad(real_out_s, [0, 0, 1, 0]).squeeze(1)
        imag_out_s = functional.pad(imag_out_s, [0, 0, 1, 0]).squeeze(1)

        out_specs_s = power_uncompress(real_out_s, imag_out_s)
        outputs_s = self.cistft(out_specs_s)

        outputs = (outputs_m + outputs_s) / 2
        output_specs = self.cstft(outputs)
        output_real, output_imag = power_compress(output_specs, cut_len=self.fft_half)
        output_mag = torch.sqrt(output_real ** 2 + output_imag ** 2)
        return output_real, output_imag, output_mag


    def test_interface(self, real, imag, srl_features):
        real = real.unsqueeze(1)[:, :, 1:]
        imag = imag.unsqueeze(1)[:, :, 1:]

        # input layer
        hx_r, hx_i = self.input_layer(real, imag)

        # encoder stages
        hx1r, hx1i = self.en1(hx_r, hx_i)
        hx2r, hx2i = self.en2(hx1r, hx1i)
        hx3r, hx3i = self.en3(hx2r, hx2i)
        hx4r, hx4i = self.en4(hx3r, hx3i)
        hx5r, hx5i = self.en5(hx4r, hx4i)
        hx6r, hx6i = self.en6(hx5r, hx5i)

        srl_features = srl_features.permute(0, 2, 1)
        srl_features = functional.interpolate(srl_features, hx6r.size(-1), mode='linear')

        out_r = rearrange(hx6r, 'b c f t -> b (c f) t')
        out_r = self.norm_r(self.conv1x1r(torch.cat([out_r, srl_features], dim=1)))
        out_i = rearrange(hx6i, 'b c f t -> b (c f) t')
        out_i = self.norm_i(self.conv1x1i(torch.cat([out_i, srl_features], dim=1)))
        out_r = rearrange(out_r, 'b (c f) t -> b c f t', f=4)
        out_i = rearrange(out_i, 'b (c f) t -> b c f t', f=4)

        # dilated dense block
        out_r, out_i = self.bottleneck(out_r, out_i)
        srl_guided_r, srl_guided_i = out_r.clone().detach(), out_i.clone().detach()

        # decoder stages - masking
        out_r_m, out_i_m = self.de1m(torch.cat([out_r, hx6r], dim=1), torch.cat([out_i, hx6i], dim=1))
        out_r_m, out_i_m = self.de2m(torch.cat([out_r_m, hx5r], dim=1), torch.cat([out_i_m, hx5i], dim=1))
        out_r_m, out_i_m = self.de3m(torch.cat([out_r_m, hx4r], dim=1), torch.cat([out_i_m, hx4i], dim=1))
        out_r_m, out_i_m = self.de4m(torch.cat([out_r_m, hx3r], dim=1), torch.cat([out_i_m, hx3i], dim=1))
        out_r_m, out_i_m = self.de5m(torch.cat([out_r_m, hx2r], dim=1), torch.cat([out_i_m, hx2i], dim=1))
        out_r_m, out_i_m = self.de6m(torch.cat([out_r_m, hx1r], dim=1), torch.cat([out_i_m, hx1i], dim=1))

        # decoder stages - mapping
        out_r_s, out_i_s = self.de1s(torch.cat([out_r, hx6r], dim=1), torch.cat([out_i, hx6i], dim=1))
        out_r_s, out_i_s = self.de2s(torch.cat([out_r_s, hx5r], dim=1), torch.cat([out_i_s, hx5i], dim=1))
        out_r_s, out_i_s = self.de3s(torch.cat([out_r_s, hx4r], dim=1), torch.cat([out_i_s, hx4i], dim=1))
        out_r_s, out_i_s = self.de4s(torch.cat([out_r_s, hx3r], dim=1), torch.cat([out_i_s, hx3i], dim=1))
        out_r_s, out_i_s = self.de5s(torch.cat([out_r_s, hx2r], dim=1), torch.cat([out_i_s, hx2i], dim=1))
        out_r_s, out_i_s = self.de6s(torch.cat([out_r_s, hx1r], dim=1), torch.cat([out_i_s, hx1i], dim=1))

        # output layer
        real_out_m, imag_out_m = self.output_layer_m(out_r_m, out_i_m)
        real_out_s, imag_out_s = self.output_layer_s(out_r_s, out_i_s)

        mask_mags = torch.sqrt(real_out_m ** 2 + imag_out_m ** 2)
        phase_real = real_out_m / (mask_mags + 1e-8)
        phase_imag = imag_out_m / (mask_mags + 1e-8)
        mask_phase = torch.atan2(phase_imag, phase_real)

        mask_mags = torch.tanh(mask_mags)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase = torch.atan2(imag, real)

        mag_out = mag * mask_mags
        phase_out = phase + mask_phase

        real_out_m = mag_out * torch.cos(phase_out)
        imag_out_m = mag_out * torch.sin(phase_out)

        real_out_m = functional.pad(real_out_m, [0, 0, 1, 0]).squeeze(1)
        imag_out_m = functional.pad(imag_out_m, [0, 0, 1, 0]).squeeze(1)

        out_specs_m = power_uncompress(real_out_m, imag_out_m)
        outputs_m = self.cistft(out_specs_m)

        real_out_s = functional.pad(real_out_s, [0, 0, 1, 0]).squeeze(1)
        imag_out_s = functional.pad(imag_out_s, [0, 0, 1, 0]).squeeze(1)

        out_specs_s = power_uncompress(real_out_s, imag_out_s)
        outputs_s = self.cistft(out_specs_s)

        outputs = (outputs_m + outputs_s) / 2
        output_specs = self.cstft(outputs)
        output_real, output_imag = power_compress(output_specs, cut_len=self.fft_half)
        output_mag = torch.sqrt(output_real ** 2 + output_imag ** 2)
        return output_real, output_imag, output_mag, srl_guided_r, srl_guided_i
