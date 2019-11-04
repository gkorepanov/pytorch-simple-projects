import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import logging

from typing import Optional, Callable, List, Tuple

logger = logging.getLogger(__name__)


def make_train_step(model, loss_fn, optimizer) -> Callable:
    """Builds function that performs a step in the train loop"""

    def train_step():
        model.train()
        e = model()
        loss = loss_fn(e)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    return train_step


def distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """Calculate distances bettween all pairs of input vectors.
    
    See https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    for some ideas about implementation.

    Args:
        x: tensor of shape Nx2 where N is the amount of vectors
            and 2 is dimensionality of each vector
    Returns:
        Matrix of NxN shape with pairwise distances"""

    diffs = x.unsqueeze(1) - x.unsqueeze(0)
    squared_diffs = diffs**2
    squared_distances = squared_diffs.sum(-1)
    squared_distances[squared_distances == 0] = 1
    return squared_distances ** 0.5


class EnergyCalculator(nn.Module):
    """PyTorch layer to calculate energy of a system of repulsive particles.

    Some of the particles are connected with springs, which
    is defined by adjacency matrix.
    Particles coordinates are stored as layer inner parameter `x`.
    
    Args:
        x: initial setting for particles coordinates, optional"""

    def __init__(self, *, adj_matrix: np.ndarray, x: Optional[np.ndarray] = None):
        super().__init__()
        
        self.N = adj_matrix.shape[0]
        
        if x is None:
            x = torch.randn(self.N, 2)
        
        self.x = nn.Parameter(x, requires_grad=True)
        
        assert adj_matrix.shape == (self.N, self.N)
        self._adj_matrix = nn.Parameter(adj_matrix, requires_grad=False)

        # mask with zeros on main diagonal
        self._mask = torch.ones((self.N, self.N)) - torch.eye(self.N)
          
    def forward(self) -> torch.Tensor:
        """Returns scalar - system energy approximation"""

        dists = distance_matrix(self.x)
        E_gravity = self._mask * 1 / dists
        E_springs = self._mask * self._adj_matrix * ((dists - 1)**2)
        E = E_gravity.sum() + E_springs.sum()
        return E


def force_layout(adj_matrix: np.ndarray,
                 lr: float = 1e-3,
                 n_iterations: int = 10000,
                 dump_each_n_iterations: int = 50) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """Generate force layout for given graph using Pytorch optimization"""

    assert (adj_matrix == adj_matrix.T).all()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adj_matrix = torch.FloatTensor(adj_matrix)

    model = EnergyCalculator(adj_matrix=adj_matrix).to(device)
    loss_fn = lambda x: (x**2).sum()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_step = make_train_step(model, loss_fn, optimizer)
    
    losses = []
    xs = []

    for i in range(n_iterations):
        model.train()
        loss = train_step()

        if i % dump_each_n_iterations:
            continue

        logger.info(f"Iteration: {i} Loss: {loss}")
        losses.append(loss)
        xs.append(model.x.detach().numpy().copy())

    return xs[-1], xs, losses

