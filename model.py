import torch.nn as nn


class FF(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output, num_layers,
                 activation='tanh', dropout_rate=0, layer_norm=False,
                 residual_connection=False):
        super(FF, self).__init__()
        assert (not residual_connection) or (dim_hidden == dim_input)
        self.residual_connection = residual_connection

        self.stack = nn.ModuleList()
        for l in range(num_layers):
            layer = []

            if layer_norm:
                layer.append(nn.LayerNorm(dim_input if l == 0 else dim_hidden))

            layer.append(nn.Linear(dim_input if l == 0 else dim_hidden,
                                   dim_hidden))
            layer.append({'tanh': nn.Tanh(), 'relu': nn.ReLU()}[activation])
            layer.append(nn.Dropout(dropout_rate))

            self.stack.append(nn.Sequential(*layer))

        self.out = nn.Linear(dim_input if num_layers < 1 else dim_hidden,
                             dim_output)

    def forward(self, x):
        for layer in self.stack:
            x = x + layer(x) if self.residual_connection else layer(x)
        return self.out(x)