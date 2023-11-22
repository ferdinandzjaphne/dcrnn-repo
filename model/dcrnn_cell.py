import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib import utils

class DCGRUCell(nn.Module):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh', filter_type="laplacian", use_gc_for_ru=True):
        super(DCGRUCell, self).__init__()

        self._num_units = num_units
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru

        self._supports = [self._build_sparse_matrix(utils.calculate_scaled_laplacian(adj_mx))]
        
        if self._use_gc_for_ru:
            self._supports.extend([self._build_sparse_matrix(utils.calculate_random_walk_matrix(adj_mx).T) for _ in range(self._max_diffusion_step)])

        self._fc_params = self._init_layer_params('fc')
        self._gconv_params = self._init_layer_params('gconv')

    def _init_layer_params(self, layer_type):
        layer_params = nn.Module()
        layer_params.weights = nn.ParameterDict()
        layer_params.biases = nn.ParameterDict()
        return layer_params

    def _build_sparse_matrix(self, L):
        L = L.tocoo()
        indices = torch.tensor(np.column_stack((L.row, L.col)), dtype=torch.long)
        values = torch.tensor(L.data, dtype=torch.float32)
        return torch.sparse.FloatTensor(indices.t().contiguous(), values, torch.Size(L.shape))

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        weights = self._fc_params.weights.get(str(inputs_and_state.shape[-1]), None)
        if weights is None:
            weights = nn.Parameter(torch.empty(inputs_and_state.shape[-1], output_size))
            nn.init.xavier_normal_(weights)
            self._fc_params.weights[str(inputs_and_state.shape[-1])] = weights
        biases = self._fc_params.biases.get(str(output_size), None)
        if biases is None:
            biases = nn.Parameter(torch.empty(output_size))
            nn.init.constant_(biases, bias_start)
            self._fc_params.biases[str(output_size)] = biases
        return F.sigmoid(torch.matmul(inputs_and_state, weights) + biases)

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        x = inputs.permute(1, 2, 0)
        x = torch.reshape(x, (self._num_nodes, -1))
        x0 = x.unsqueeze(0)
        x = x0
        for support in self._supports:
            x1 = torch.sparse.mm(support, x0)
            x = torch.cat([x, x1], dim=0)
            for _ in range(2, self._max_diffusion_step + 1):
                x2 = 2 * torch.sparse.mm(support, x1) - x0
                x = torch.cat([x, x2], dim=0)
                x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        x = torch.reshape(x, (num_matrices, self._num_nodes, -1)).permute(2, 0, 1)
        x = torch.reshape(x, (-1, x.shape[-1]))

        weights = self._gconv_params.weights.get(str(x.shape[-1]), None)
        if weights is None:
            weights = nn.Parameter(torch.empty(x.shape[-1], output_size))
            nn.init.xavier_normal_(weights)
            self._gconv_params.weights[str(x.shape[-1])] = weights
        biases = self._gconv_params.biases.get(str(output_size), None)
        if biases is None:
            biases = nn.Parameter(torch.empty(output_size))
            nn.init.constant_(biases, bias_start)
            self._gconv_params.biases[str(output_size)] = biases
        return torch.matmul(x, weights) + biases

    def forward(self, inputs, hx):
        output_size = 2 * self._num_units
        fn = self._gconv if self._use_gc_for_ru else self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        r, u = torch.split(value, self._num_units, dim=-1)
        c = self._gconv(inputs, r * hx, self._num_units)
        new_state = u * hx + (1.0 - u) * F.tanh(c)
        return new_state