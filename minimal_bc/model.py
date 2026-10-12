import torch
from torch import nn

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
    def __init__(self, obs_dim, act_dim, hidden_dim, depth):
        super().__init__()
        input_dim = obs_dim + act_dim + 1
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(depth-1)
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, act_dim))

    def forward(self, obs, xt, t): # velocity field model
        # TODO: t encoding
        t = t
        inputs = torch.cat([obs, xt, t], dim=1)
        h = self.inp(inputs)
        for block in self.blocks:
            h = h + block(h)
        out = self.output(h)
        return out

if __name__ == '__main__':
    model = BCModel()
    out = model(torch.randn(4, 28))
    assert out.shape == (4, 7)
    print(model)
    print('params', sum(p.numel() for p in model.parameters()))
