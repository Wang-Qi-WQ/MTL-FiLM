import math
from typing import Tuple

import torch
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Permute(nn.Module):
    def __init__(self, target_dims: Tuple):
        super().__init__()
        self.td = target_dims

    def forward(self, x):
        return x.permute(*self.td)


class ViewAtLast2Dim(nn.Module):
    def __init__(self, last2dim, last1dim):
        super().__init__()
        self.l2 = last2dim
        self.l1 = last1dim

    def forward(self, x):
        assert len(x.shape) == 3
        b, t, f = x.shape
        x = x.view(b, t, self.l2, self.l1)
        return x


class FGbiLSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=64, bi_dir=True, proj_size=None):
        nn.Module.__init__(self)
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                               batch_first=True, bidirectional=bi_dir)
        self.proj_size = input_size if proj_size is None else proj_size
        self.d = 2 if bi_dir else 1
        self.fc = nn.Linear(hidden_size * self.d, self.proj_size)

    def forward(self, x):
        bs, ch, fi, pn = x.size()
        x = x.transpose(1, 3).reshape(bs * pn, fi, ch)
        self.bi_lstm.flatten_parameters()
        x = self.bi_lstm(x)[0]
        x = self.fc(x.reshape(bs, pn, fi, -1)).transpose(1, 3)
        return x


class CNNTrainableKernelChannelToOneGroupMaskVer7(nn.Module):
    """
        1. #Codes, to catch and cast gamma & beta
    """

    def __init__(self, freq: int = 352, bins_per_octave: int = 48, heads: int = 64, concat_heads: int = None,
                 lstm_cells: int = 16, dp: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(heads, 1, 2 * freq - 1))
        self.bias = nn.Parameter(torch.zeros(heads))
        self.scaling = nn.Parameter(torch.zeros(freq))

        self.linear_inp = nn.Conv1d(heads, 6 * heads, 1)
        self.linear_g = nn.Conv1d(3 * heads if concat_heads is None else 3 * concat_heads, heads, 1)
        self.linear_b = nn.Conv1d(3 * heads if concat_heads is None else 3 * concat_heads, heads, 1)
        self.bn_g = nn.BatchNorm1d(heads)
        self.bn_b = nn.BatchNorm1d(heads)
        self.dropout_1 = nn.Dropout(p=dp)
        self.dropout_2 = nn.Dropout(p=dp)

        self.fglstm_g = FGbiLSTM(input_size=heads, hidden_size=lstm_cells)
        self.fglstm_b = FGbiLSTM(input_size=heads, hidden_size=lstm_cells)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        return

    def forward(self, x, g_in=None, b_in=None):
        b, c, t, f = x.shape
        q = x.transpose(1, 2).reshape(b * t, c, f)
        # q: [b * t, c, f]
        z = torch.nn.functional.conv1d(q, self.weight, self.bias, padding='same', groups=c)
        z *= self.scaling
        # z: [b * t, c, f]

        gamma, beta = nn.functional.gelu(self.linear_inp(z)).chunk(2, 1)
        # gamma: [b * t, 3c, f], beta: [b * t, 3c, f]
        if g_in is not None and b_in is not None:
            gamma = self.bn_g(self.linear_g(torch.cat([gamma, g_in], dim=1))).reshape(b, t, c, f).transpose(1, 2)
            beta = self.bn_b(self.linear_b(torch.cat([beta, b_in], dim=1))).reshape(b, t, c, f).transpose(1, 2)

            # gamma: [b, c, t, f], beta: [b, c, t, f]
            gamma = self.dropout_1(self.fglstm_g(gamma))
            beta = self.dropout_2(self.fglstm_b(beta))

            return x + gamma * x + beta, None, None

        else:
            g_out, b_out = gamma.detach(), beta.detach()
            gamma = self.bn_g(self.linear_g(gamma)).reshape(b, t, c, f).transpose(1, 2)
            beta = self.bn_b(self.linear_b(beta)).reshape(b, t, c, f).transpose(1, 2)

            # gamma: [b, c, t, f], beta: [b, c, t, f]
            gamma = self.dropout_1(self.fglstm_g(gamma))
            beta = self.dropout_2(self.fglstm_b(beta))

            return x + gamma * x + beta, g_out, b_out


class MLPOut(nn.Module):
    def __init__(self, heads_inp: int = 64, heads_out: int = None):
        super().__init__()
        self.mlp_out = nn.Sequential(*[
            nn.Conv2d(heads_inp, 2 * heads_inp, 1),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Conv2d(2 * heads_inp, 1 if heads_out is None else heads_out, 1)])

    def forward(self, x):
        # x: [b, c, t, f]
        return self.mlp_out(x).squeeze(1)


class InpBlock(nn.Module):
    def __init__(self, spectrum: int = 352, freq: int = 88,
                 onset_heads: int = 64, frame_heads: int = 32, velocity_heads: int = 64):
        super().__init__()
        self.spectrum = spectrum
        self.freq = freq
        self.g = spectrum // freq

        self.bn_inp = nn.Sequential(*[Transpose(1, 2), nn.BatchNorm1d(spectrum), Transpose(1, 2)])
        self.onset_inp = nn.Sequential(*[Permute((0, 3, 1, 2)), nn.Conv2d(self.g, onset_heads, 1)])
        self.frame_inp = nn.Sequential(*[Permute((0, 3, 1, 2)), nn.Conv2d(self.g, frame_heads, 1)])
        self.velocity_inp = nn.Sequential(*[Permute((0, 3, 1, 2)), nn.Conv2d(self.g, velocity_heads, 1)])

    def forward(self, x):
        b, t, f = x.shape
        x = self.bn_inp(x)
        # x: [b, t, 352]
        x = x.view(b, t, self.freq, self.g)
        # x: [b, t, 88, 4]
        onset = self.onset_inp(x)
        frame = self.frame_inp(x)
        velocity = self.velocity_inp(x)
        # *: [b, c, t, 88]
        return onset, frame, velocity


class InpBlockPedalVer6(nn.Module):

    def __init__(self, spectrum: int = 352, freq: int = 88, pedal_heads: int = 32, sub_pedal_heads: int = 32):
        super().__init__()
        self.spectrum = spectrum
        self.freq = freq
        self.g = spectrum // freq

        self.inp_main = nn.Sequential(*[Transpose(1, 2), nn.BatchNorm1d(spectrum), Transpose(1, 2),
                                        ViewAtLast2Dim(self.freq, self.g),
                                        Permute((0, 3, 1, 2)), nn.Conv2d(self.g, pedal_heads, 1)])
        self.inp_other = nn.Sequential(*[Transpose(1, 2), nn.BatchNorm1d(spectrum), Transpose(1, 2),
                                       ViewAtLast2Dim(self.freq, self.g),
                                       Permute((0, 3, 1, 2)), nn.Conv2d(self.g, sub_pedal_heads, 1)])

    def forward(self, x):
        # x: [b, t, 352]
        x_main = self.inp_main(x)
        x_other = self.inp_other(x)
        # *: [b, c, t, 88]
        x = (x_main, x_other)
        return x


class StemBlock(nn.Module):
    def __init__(self, freq: int = 88,
                 onset_heads: int = 64, onset_cells: int = 16,
                 frame_heads: int = 32, frame_cells: int = 32,
                 velocity_heads: int = 64, velocity_cells: int = 32):
        super().__init__()
        self.onset_stem = CNNTrainableKernelChannelToOneGroupMaskVer7(
            freq=freq, bins_per_octave=12, heads=onset_heads, lstm_cells=onset_cells)
        self.frame_stem = CNNTrainableKernelChannelToOneGroupMaskVer7(
            freq=freq, bins_per_octave=12, heads=frame_heads, concat_heads=onset_heads + frame_heads,
            lstm_cells=frame_cells)
        self.velocity_stem = CNNTrainableKernelChannelToOneGroupMaskVer7(
            freq=freq, bins_per_octave=12, heads=velocity_heads, concat_heads=onset_heads + velocity_heads,
            lstm_cells=velocity_cells)

    def forward(self, x):
        x_ons, x_frm, x_vel = x
        # *: [b, c, t, 88]
        x_ons, g, b = self.onset_stem(x_ons, None, None)
        x_frm, _, _ = self.frame_stem(x_frm, g, b)
        x_vel, _, _ = self.velocity_stem(x_vel, g, b)
        # *: [b, c, t, 88]
        x = (x_ons, x_frm, x_vel)
        return x


class StemBlockPedalVer7(nn.Module):

    def __init__(self, freq: int = 88,
                 pedal_heads: int = 32, pedal_cells: int = 32,
                 sub_pedal_heads: int = 16, sub_pedal_cells: int = 16):
        super().__init__()
        self.stem_main = CNNTrainableKernelChannelToOneGroupMaskVer7(
            freq=freq, heads=pedal_heads, lstm_cells=pedal_cells,
            concat_heads=pedal_heads + sub_pedal_heads)
        self.stem_other = CNNTrainableKernelChannelToOneGroupMaskVer7(
            freq=freq, heads=sub_pedal_heads, lstm_cells=sub_pedal_cells)

    def forward(self, x):
        # *: [b, c, t, 88]
        x_main, x_other = x
        x_other, g_other, b_other = self.stem_other(x_other, None, None)
        x_main, _, _ = self.stem_main(x_main, g_other, b_other)
        # *: [b, c, t, 88]
        x = (x_main, x_other)
        return x


class OutBlock(nn.Module):
    def __init__(self, onset_heads: int = 64, frame_heads: int = 32, velocity_heads: int = 64):
        super().__init__()
        self.onset_out = MLPOut(heads_inp=onset_heads)
        self.frame_out = MLPOut(heads_inp=frame_heads, heads_out=2)
        self.velocity_out = nn.Sequential(*[MLPOut(heads_inp=velocity_heads, heads_out=128)])

    def forward(self, onset, frame, velocity):
        onset = self.onset_out(onset)
        frame, offset = self.frame_out(frame).chunk(2, 1)
        velocity = self.velocity_out(velocity)
        return torch.sigmoid(onset), torch.sigmoid(frame.squeeze(1)), torch.sigmoid(offset.squeeze(1)), velocity


class OutBlockPedalVer9(nn.Module):

    def __init__(self, pedal_heads: int = 32, freq: int = 88, sub_pedal_heads: int = 16):
        super().__init__()
        self.frm_out = nn.Sequential(*[MLPOut(heads_inp=pedal_heads, heads_out=2), nn.Linear(freq, 1)])
        self.vel_out = nn.Sequential(*[MLPOut(heads_inp=sub_pedal_heads, heads_out=128), nn.Linear(freq, 1)])

    def forward(self, x):
        x_main, x_vel = x
        frm, off = self.frm_out(x_main).squeeze(-1).chunk(2, 1)
        vel = self.vel_out(x_vel).squeeze(-1)

        return torch.sigmoid(frm.squeeze(1)), torch.sigmoid(off.squeeze(1)), vel


class MT_FiLM(nn.Module):
    """
    [Ver17 <- Ver12, Ver3-Abl3] Diff:
    1. All branches
    2. #Codes, add a branch-cross operator & others
    model params: 1234883
    """

    def __init__(self, spectrum: int = 352, freq: int = 88, n_blocks: int = 4,
                 onset_heads: int = 64, onset_cells: int = 16,
                 frame_heads: int = 32, frame_cells: int = 32,
                 velocity_heads: int = 64, velocity_cells: int = 32):
        super().__init__()

        self.inp_block = InpBlock(
            spectrum=spectrum, freq=freq,
            onset_heads=onset_heads, frame_heads=frame_heads, velocity_heads=velocity_heads
        )
        self.stem_blocks = nn.Sequential(*[StemBlock(
            freq=freq, onset_heads=onset_heads, onset_cells=onset_cells,
            frame_heads=frame_heads, frame_cells=frame_cells,
            velocity_heads=velocity_heads, velocity_cells=velocity_cells)
            for _ in range(n_blocks)])
        self.out_block = OutBlock(
            onset_heads=onset_heads, frame_heads=frame_heads, velocity_heads=velocity_heads
        )

    def forward(self, x):
        # x: [b, t, 352]
        onset, frame, velocity = self.inp_block(x)
        # *: [b, c, t, 88]
        onset, frame, velocity = self.stem_blocks((onset, frame, velocity))
        # *: [b, c, t, 88]
        onset, frame, offset, velocity = self.out_block(onset, frame, velocity)
        # *: [b, (128,) t, 88]
        return frame, onset, offset, velocity


class MT_FiLMSustainPedal(nn.Module):

    def __init__(self, spectrum: int = 352, freq: int = 88, n_blocks: int = 4,
                 pedal_heads: int = 64, pedal_cells: int = 64, sub_pedal_heads: int = 16, sub_pedal_cells: int = 16):
        super().__init__()

        self.inp_block = InpBlockPedalVer6(spectrum=spectrum, freq=freq, pedal_heads=pedal_heads,
                                           sub_pedal_heads=sub_pedal_heads)
        self.stem_blocks = nn.Sequential(*[StemBlockPedalVer7(
            freq=freq, pedal_heads=pedal_heads, pedal_cells=pedal_cells,
            sub_pedal_heads=sub_pedal_heads, sub_pedal_cells=sub_pedal_cells
        ) for _ in range(n_blocks)])
        self.out_block = OutBlockPedalVer9(pedal_heads=pedal_heads, freq=freq, sub_pedal_heads=sub_pedal_heads)

    def forward(self, x):
        # x: [b, t, 352]
        x = self.inp_block(x)
        # *: [b, c, t, 88]
        x = self.stem_blocks(x)
        # *: [b, c, t, 88]
        frm, off, vel = self.out_block(x)
        # *: [b, (128,) t]
        return frm, None, off, vel, None, None, None, None, None, None, None, None


