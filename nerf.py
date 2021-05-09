import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

class NerfTransform(torch.nn.Module):
    def __init__(self, num_input_channels, max_freq=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self.freq_bands = 2.**torch.linspace(0.,max_freq-1 , steps=max_freq)
        self.fn = [torch.sin, torch.cos]
    def forward(self, x):

        num_pts, channels = x.shape
        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)
        embed_fns = []
        for freq in self.freq_bands:
            for p_fn in self.fn:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
        x = torch.cat([fn(2*math.pi*x) for fn in embed_fns], -1)
        return x



class Nerf(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 input_ch=3,
                 output_ch=3,
                 skips=[4]):
        """
        """
        super(Nerf, self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] + [
            nn.Linear(W, W) if i not in
            self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)
        ])
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        outputs = self.output_linear(h)
        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1]))


