import torch

def lb_loss_func(
    gate_softmax: torch.Tensor, 
    num_experts: int, 
    top_k: int = 2
) -> float:
    _, selected_experts = torch.topk(gate_softmax, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, 
        num_classes=num_experts
    )
    expert_mask = torch.max(expert_mask, dim=-2).values.float()
    tokens_per_expert = torch.mean(expert_mask, dim=0)
    router_prob_per_expert = torch.mean(gate_softmax, dim=0)
    return torch.sum(tokens_per_expert * router_prob_per_expert) * num_experts