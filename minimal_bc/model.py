import math
import torch
from torch import nn


def time_embedding(t, dim):
    # Sinusoidal features of t in [0, 1], as for diffusion timesteps (DiT):
    # [sin(1000 t w_k), cos(1000 t w_k)], w_k = 10000^(-k / (dim/2)), k = 0 .. dim/2 - 1
    freqs = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    angles = 1000 * t * freqs  # (B, dim/2)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class BCModel(nn.Module):
    # residual: False = plain MLP; True = skip connection around each hidden layer of the plain MLP
    # (post-LN, exp009); 'pre_ln' = pre-LN residual MLP (always uses LayerNorm).
    def __init__(self, obs_dim=28, act_dim=7, hidden_dim=512, depth=3, layer_norm=True, residual=False):
        super().__init__()
        self.residual = residual
        if residual == 'pre_ln':
            # Input projection, then h = h + Linear(ReLU(LayerNorm(h))) for the remaining hidden layers,
            # and a final LayerNorm before the output layer. Same number of Linear layers as the plain MLP.
            self.inp = nn.Linear(obs_dim, hidden_dim)
            self.blocks = nn.ModuleList(
                nn.Sequential(nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                for _ in range(depth - 1))
            self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, act_dim))
            return

        # a pure MLP with layers
        layers = []
        in_dim = obs_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))  # keeps hidden activations at a stable scale
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)
        self.layer_size = 3 if layer_norm else 2  # modules per hidden layer: Linear, ReLU(, LayerNorm)

    def forward(self, obs):
        if self.residual == 'pre_ln':
            h = self.inp(obs)
            for block in self.blocks:
                h = h + block(h)
            return self.out(h)
        if not self.residual:
            return self.net(obs)
        # Same layers and parameters, plus a skip connection h + f(h) around every hidden layer
        # except the first, which changes the width (obs_dim -> hidden_dim).
        h = obs
        for i in range(0, len(self.net) - 1, self.layer_size):
            out = self.net[i:i + self.layer_size](h)
            h = out if i == 0 else h + out
        return self.net[-1](h)

class BCFlowModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, depth, film=False, t_embed_dim=0, cond_input=True):
        super().__init__()
        # t_embed_dim: 0 = feed t as a raw scalar; > 0 = feed t_embed_dim sinusoidal features of t
        self.t_embed_dim = t_embed_dim
        t_dim = t_embed_dim or 1
        # cond_input: concatenate [obs, t] with x_t at the input; False = the input is x_t only,
        # and obs and t enter only through FiLM (standard FiLM conditioning, needs film=True)
        self.cond_input = cond_input
        input_dim = act_dim + (obs_dim + t_dim if cond_input else 0)
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(depth-1)
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, act_dim))
        # film: [obs, t] also gives a scale and shift after the LayerNorm of every block.
        # Zero-initialized, so training starts from the same function as without FiLM.
        self.film = None
        if film:
            self.film = nn.ModuleList(nn.Linear(obs_dim + t_dim, 2 * hidden_dim) for _ in range(depth - 1))
            for layer in self.film:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, obs, xt, t): # velocity field model
        if self.t_embed_dim:
            t = time_embedding(t, self.t_embed_dim)
        h = self.inp(torch.cat([obs, xt, t], dim=1) if self.cond_input else xt)
        cond = torch.cat([obs, t], dim=1)
        for i, block in enumerate(self.blocks):
            if self.film is None:
                h = h + block(h)
            else:
                scale, shift = self.film[i](cond).chunk(2, dim=1)
                z = block[0](h) * (1 + scale) + shift  # LayerNorm, then FiLM
                h = h + block[2](block[1](z))  # ReLU, Linear
        out = self.output(h)
        return out

if __name__ == '__main__':
    model = BCModel()
    out = model(torch.randn(4, 28))
    assert out.shape == (4, 7)
    print(model)
    print('params', sum(p.numel() for p in model.parameters()))
