# PyStemma

PyStemma: A PyTorch-based library for Spatio-Temporal Explainable Modular Models and Architectures.

A PyTorch-based library for building modular and explainable spatio-temporal models and architectures.

## Installation

PyStemma requires Python 3.10 or newer and PyTorch 2.9.1 or newer.

Install PyStemma from PyPI:

```bash
python -m pip install pystemma
```

## Example

The following example shows how a PyStemma component can be combined with standard PyTorch operations.

```python
import torch

from pystemma.nn.graph.masking import GlobalEdgeMask

mask_generator = GlobalEdgeMask(num_nodes=3)

adjacency = torch.tensor(
    [
        [0.0, 1.0, 0.5],
        [0.2, 0.0, 1.0],
        [1.0, 0.3, 0.0],
    ]
)
node_features = torch.randn(3, 2)

edge_mask = mask_generator()
masked_adjacency = adjacency * edge_mask
aggregated_features = masked_adjacency @ node_features

loss = aggregated_features.square().mean()
loss.backward()
```
