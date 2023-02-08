import torch
import config

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
    out = nn_policy * target_policy
    return -out.sum(dim=1).mean()


def value_loss(nn_value: torch.Tensor, real_value: torch.Tensor):
    return (nn_value - real_value).pow(2).mean()


def loss_fn(nn_val, nn_pol, target_val, target_pol) -> torch.Tensor:
    return (
        config.value_loss_ratio * value_loss(nn_val, target_val) 
        +policy_loss(nn_pol, target_pol)
    )

