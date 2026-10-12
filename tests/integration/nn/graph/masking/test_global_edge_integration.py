import torch

from pystemma.nn.graph.masking import GlobalEdgeMask


def test_external_support_application_updates_scores_with_task_loss():
    mask_generator = GlobalEdgeMask(num_nodes=2, initial_value=0.1)
    support = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    node_features = torch.tensor([[1.0], [2.0]])
    target = torch.zeros(2, 1)
    optimizer = torch.optim.SGD(mask_generator.parameters(), lr=0.01)
    scores_before = mask_generator.scores.detach().clone()

    optimizer.zero_grad()
    edge_mask = mask_generator()
    masked_support = support * edge_mask
    prediction = masked_support @ node_features
    task_loss = torch.nn.functional.mse_loss(prediction, target)
    task_loss.backward()

    assert mask_generator.scores.grad is not None
    assert torch.isfinite(mask_generator.scores.grad).all()
    assert torch.any(mask_generator.scores.grad != 0)

    optimizer.step()

    assert not torch.equal(mask_generator.scores, scores_before)
