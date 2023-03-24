import torch
from config import config

def policy_loss(
    nn_policy: torch.Tensor, 
    target_policy: torch.Tensor
):
    """
    nn output is 
        batch x policy_dim
    assumed that logsoftmax has been applied

    search_result is
        batch x policy_dim
    """

    assert target_policy.sum(dim=1).allclose(torch.ones(nn_policy.shape[0], device=target_policy.device))

    assert (target_policy >= 0).all()

    out = nn_policy * target_policy

    return -out.sum(dim=1)


def value_loss(nn_value: torch.Tensor, real_value: torch.Tensor):
    return (nn_value - real_value).pow(2)


def loss_fn(nn_val:torch.Tensor, nn_pol:torch.Tensor, target_val:torch.Tensor, target_pol:torch.Tensor, weights:torch.Tensor) -> torch.Tensor:

    val_loss = value_loss(nn_val, target_val)
    pol_loss = policy_loss(nn_pol, target_pol)

    val_loss = (weights * val_loss).mean()
    pol_loss = (weights * pol_loss).mean()

    return (config['value_loss_ratio'] * val_loss + pol_loss)

