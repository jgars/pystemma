import pytest
import torch

from pystemma.nn.graph.masking import GlobalEdgeMask


@pytest.mark.parametrize(
    "invalid_num_nodes",
    [True, False, 2.0, "2", None, torch.tensor(2)],
)
def test_constructor_rejects_invalid_num_nodes_types(invalid_num_nodes):
    with pytest.raises(TypeError, match="num_nodes"):
        GlobalEdgeMask(invalid_num_nodes)


@pytest.mark.parametrize("invalid_num_nodes", [0, -1])
def test_constructor_rejects_non_positive_num_nodes(invalid_num_nodes):
    with pytest.raises(ValueError, match="num_nodes"):
        GlobalEdgeMask(invalid_num_nodes)


@pytest.mark.parametrize(
    "invalid_initial_value",
    [True, False, "0.1", None, 1 + 0j, torch.tensor(0.1)],
)
def test_constructor_rejects_invalid_initial_value_types(invalid_initial_value):
    with pytest.raises(TypeError, match="initial_value"):
        GlobalEdgeMask(2, initial_value=invalid_initial_value)


@pytest.mark.parametrize("invalid_initial_value", [float("nan"), float("inf"), -float("inf")])
def test_constructor_rejects_non_finite_initial_values(invalid_initial_value):
    with pytest.raises(ValueError, match="initial_value"):
        GlobalEdgeMask(2, initial_value=invalid_initial_value)


def test_constructor_registers_one_float32_scores_parameter():
    module = GlobalEdgeMask(3)

    assert tuple(module.scores.shape) == (3, 3)
    assert module.scores.dtype == torch.float32
    assert module.scores.requires_grad
    assert list(module.named_parameters()) == [("scores", module.scores)]
    assert list(module.state_dict()) == ["scores"]
    torch.testing.assert_close(module.scores, torch.full((3, 3), 0.1))


def test_constructor_initializes_scores_to_the_requested_value():
    module = GlobalEdgeMask(2, initial_value=-0.25)

    torch.testing.assert_close(module.scores, torch.full((2, 2), -0.25))


def test_forward_uses_strict_threshold_and_returns_exact_binary_values():
    module = GlobalEdgeMask(2)
    with torch.no_grad():
        module.scores.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0001]]))

    edge_mask = module()

    expected = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    assert edge_mask.shape == (2, 2)
    assert edge_mask.dtype == torch.float32
    assert torch.equal(edge_mask, expected)
    assert set(edge_mask.unique().tolist()) <= {0.0, 1.0}


def test_forward_preserves_directed_asymmetric_and_diagonal_values():
    module = GlobalEdgeMask(3)
    scores = torch.tensor(
        [
            [1.0, -1.0, 0.5],
            [1.0, 0.0, -0.5],
            [-0.5, 0.25, 1.0],
        ]
    )
    with torch.no_grad():
        module.scores.copy_(scores)

    edge_mask = module()

    expected = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    assert torch.equal(edge_mask, expected)
    assert not torch.equal(edge_mask, edge_mask.transpose(0, 1))
    assert torch.equal(torch.diagonal(edge_mask), torch.tensor([1.0, 0.0, 1.0]))


def test_forward_accepts_no_graph_or_batch_input():
    module = GlobalEdgeMask(2)

    with pytest.raises(TypeError):
        module(torch.ones(2, 2))


def test_backward_returns_the_upstream_gradient_for_scores():
    module = GlobalEdgeMask(3)
    with torch.no_grad():
        module.scores.copy_(
            torch.tensor(
                [
                    [-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-0.5, 0.0, 0.5],
                ]
            )
        )
    upstream = torch.tensor(
        [
            [1.5, -2.0, 0.75],
            [0.0, 3.25, -1.25],
            [2.5, -0.5, 4.0],
        ]
    )

    edge_mask = module()
    loss = (edge_mask * upstream).sum()
    loss.backward()

    # The identity STE propagates the upstream gradient unchanged across the hard threshold.
    torch.testing.assert_close(module.scores.grad, upstream)


def test_external_task_and_sparsity_gradients_add_through_the_mask():
    module = GlobalEdgeMask(2)
    support = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    target = torch.tensor([[0.5, 1.0], [0.0, 2.0]])
    regularization_weight = 0.25

    edge_mask = module()
    task_loss = (support * edge_mask - target).square().mean()
    sparsity_loss = (edge_mask.mean() - 0.25).abs()

    (task_gradient,) = torch.autograd.grad(
        task_loss,
        module.scores,
        retain_graph=True,
    )
    (sparsity_gradient,) = torch.autograd.grad(
        sparsity_loss,
        module.scores,
        retain_graph=True,
    )
    (combined_gradient,) = torch.autograd.grad(
        task_loss + regularization_weight * sparsity_loss,
        module.scores,
    )

    torch.testing.assert_close(
        combined_gradient,
        task_gradient + regularization_weight * sparsity_gradient,
    )
    expected_task_gradient = torch.tensor(
        [
            [1.5, 1.0],
            [0.125, 1.5],
        ]
    )
    expected_sparsity_gradient = torch.full((2, 2), 0.25)
    expected_combined_gradient = torch.tensor(
        [
            [1.5625, 1.0625],
            [0.1875, 1.5625],
        ]
    )

    torch.testing.assert_close(task_gradient, expected_task_gradient)
    torch.testing.assert_close(sparsity_gradient, expected_sparsity_gradient)
    torch.testing.assert_close(combined_gradient, expected_combined_gradient)


def test_train_and_eval_modes_are_deterministic_and_equivalent():
    module = GlobalEdgeMask(3, initial_value=-0.1)
    with torch.no_grad():
        module.scores[0, 0] = 0.2

    module.train()
    train_edge_mask = module()
    module.eval()
    eval_edge_mask = module()

    assert torch.equal(train_edge_mask, eval_edge_mask)
    assert not module.training


def test_state_dict_round_trip_restores_scores_and_mask():
    source = GlobalEdgeMask(3)
    with torch.no_grad():
        source.scores.copy_(torch.tensor([[1.0, -1.0, 0.0], [0.5, 0.0, -0.5], [-1.0, 1.0, 1.0]]))
    state = {name: value.clone() for name, value in source.state_dict().items()}

    restored = GlobalEdgeMask(3, initial_value=-2.0)
    load_result = restored.load_state_dict(state)

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    torch.testing.assert_close(restored.scores, source.scores)
    assert torch.equal(restored(), source())


def test_to_dtype_preserves_mask_semantics_and_changes_output_dtype():
    module = GlobalEdgeMask(2).to(dtype=torch.float64)
    with torch.no_grad():
        module.scores.copy_(torch.tensor([[-1.0, 0.0], [0.5, 1.0]], dtype=torch.float64))

    edge_mask = module()

    assert module.scores.dtype == torch.float64
    assert edge_mask.dtype == torch.float64
    assert torch.equal(edge_mask, torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_to_cuda_moves_scores_and_mask():
    module = GlobalEdgeMask(2).to(device="cuda")
    with torch.no_grad():
        module.scores.copy_(torch.tensor([[-1.0, 0.0], [0.5, 1.0]], device="cuda"))

    edge_mask = module()

    assert module.scores.device.type == "cuda"
    assert edge_mask.device.type == "cuda"
    assert torch.equal(edge_mask.cpu(), torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
