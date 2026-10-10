import torch
from torch import nn

class BCModel(nn.Module):
    def __init__(self, obs_dim=28, act_dim=7, hidden_dim=512, depth=3, layer_norm=True):
        super().__init__()
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

    def forward(self, obs):
        return self.net(obs)

if __name__ == '__main__':
    model = BCModel()
    out = model(torch.randn(4, 28))
    assert out.shape == (4, 7)
    print(model)
    print('params', sum(p.numel() for p in model.parameters()))