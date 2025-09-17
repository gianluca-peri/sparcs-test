import torch
import torch.nn as nn


class SPARCS(nn.Module):
    """
    SPARCS module implementing the spectral parameterization.
    """

    def __init__(self, layers_dim, activation="relu", bias=True):
        """
        Initialize SPARCS module.

        Args:
            layers_dim (list[int]): Dimensions of each layer.
            activation (str): Non-linearity ('relu', 'tanh', ...).
            bias (bool): If True, append bias neuron to input.
        """
        super().__init__()

        # Work on a copy of layers_dim to avoid modifying input
        self.layers_dim = layers_dim.copy()
        if bias:
            self.layers_dim[0] += 1

        # Activation lookup
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.non_lin = activations[activation]

        # Phi blocks (rectangular learnable matrices)
        self.phi_blocks = nn.ParameterList([
            nn.Parameter(torch.empty(self.layers_dim[i + 1], self.layers_dim[i]))
            for i in range(len(self.layers_dim) - 1)
        ])

        # Lambda diagonals (layer-wise spectral params)
        self.lambda_diags = nn.ParameterList([
            nn.Parameter(torch.empty(dim)) for dim in self.layers_dim
        ])

        # Initialize params
        for phi in self.phi_blocks:
            nn.init.xavier_uniform_(phi)
        for i, lam in enumerate(self.lambda_diags):
            lam.data.fill_(0. if i != len(self.lambda_diags) - 1 else 1.)

        self.bias = bias
        self.number_of_layers = len(self.layers_dim)
        self.W = None

    def reset_weights(self):
        """
        Reset the cached weight matrices.
        """
        self.W = None

    def reg_term(self, reg_cost=1e-4):
        """
        Compute the regularization term: lasso on the eigenvalues.
        Notice, the last diagonal is not regularized.
        """
        reg = 0.0
        for i, lam in enumerate(self.lambda_diags):
            if i != len(self.lambda_diags) - 1:
                reg += torch.sum(torch.abs(lam))
        return reg * reg_cost

    def build_weight_matrices(self):
        """
        Construct block weight matrices W[i][j] following the formula:

        W^{(B)}_{i,j} = 
            phi_{i-1} L_{i-1} - L_i phi_{i-1},          if i - j == 1
            (-1)^{i-1-j} [phi_{i-1} L_{i-1} - L_i phi_{i-1}] prod_{k=1}^{i-1-j} phi_{i-1-k},  if i - j > 1
        """
        W = [[None for _ in range(i)] for i in range(self.number_of_layers)]

        # Build blocks
        for i in range(1, self.number_of_layers):
            lam_i = torch.diag(self.lambda_diags[i])
            phi_im1 = self.phi_blocks[i - 1]
            lam_im1 = torch.diag(self.lambda_diags[i - 1])

            base = phi_im1 @ lam_im1 - lam_i @ phi_im1  # [dim_i, dim_{i-1}]

            # Direct connection (i-j == 1)
            W[i][i - 1] = base

            # Longer connections (i-j > 1)
            for j in range(i - 1):
                prod = base.clone()
                for k in range(1, i - j):
                    prod = prod @ self.phi_blocks[i - 1 - k]
                sign = -1 if (i - 1 - j) % 2 else 1
                W[i][j] = sign * prod

        return W
    
    def select_just_best_projectors(self, num_best=1):
        """
        Set to zero all eigenvalues except the top `num_best` ones, globally.
        """
        all_lambdas = torch.cat([lam.flatten() for lam in self.lambda_diags])
        if num_best >= len(all_lambdas):
            return  # Nothing to do

        topk_values, _ = torch.topk(all_lambdas, num_best)
        threshold = topk_values[-1].item()

        for lam in self.lambda_diags:
            lam.data = torch.where(lam.data >= threshold, lam.data, torch.zeros_like(lam.data))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Shape (batch, input_dim).
        Returns:
            torch.Tensor: Output activations of last layer.
        """
        if self.training or self.W is None:
            self.W = self.build_weight_matrices()

        if self.bias:
            ones = torch.ones(x.size(0), 1, device=x.device)
            x = torch.cat([x, ones], dim=1)

        activations = [x]

        for i in range(1, self.number_of_layers):
            a_i = activations[i - 1] @ self.W[i][i - 1].T
            for j in range(i - 1):
                a_i += activations[j] @ self.W[i][j].T

            a_i /= i
            a_i = self.non_lin(a_i)

            activations.append(a_i)

        return activations[-1]


