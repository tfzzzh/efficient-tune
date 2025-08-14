from typing import Dict, Tuple, Literal
import torch
import numpy as np
import warnings


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """
    Args:
        reward_fn (Callable[[str, str], dict[str, float]]): Scores the rollout responses against the ground truths.
            producing a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses (list[str]): Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths (list[str]): The ground truths for the examples. The length of this list is
            rollout_batch_size, the ground truth for each example is repeated group_size times.
        group_size (int): Number of responses per question (group).
        advantage_eps (float):
        normalize_by_std (bool): If True, divide by the per-group standard deviation; otherwise subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
            "advantages" shape (rollout_batch_size,). Group-normalized rewards for each rollout response
            "raw_rewards" shape (rollout_batch_size,). Unnormalized rewards for each rollout response.

            # may be I shall keep format reward
    """
    rollout_batch_size = len(rollout_responses)
    assert (
        rollout_batch_size % group_size == 0
    ), f"rollout_batch_size %% group_size = {rollout_batch_size % group_size}"

    n_prompts = rollout_batch_size // group_size
    advantages = np.empty(shape=(rollout_batch_size,))
    rewards = np.empty(shape=(rollout_batch_size,))

    for i in range(n_prompts):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size

        for j in range(start_idx, end_idx):
            reward = reward_fn(rollout_responses[j], repeated_ground_truths[j])
            rewards[j] = reward["reward"]

        rewards_g = rewards[start_idx:end_idx]
        mu = np.mean(rewards_g)
        std = np.sqrt(
            np.mean((rewards_g - mu) * (rewards_g - mu))
            * (group_size / (group_size - 1))
        )
        if normalize_by_std:
            advantages[start_idx:end_idx] = (rewards_g - mu) / (std + advantage_eps)

        else:
            advantages[start_idx:end_idx] = rewards_g - mu

    # return & cast
    results = {
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "raw_rewards": torch.tensor(rewards, dtype=torch.float32),
    }

    return results["advantages"], results["raw_rewards"], {}


def compute_advantages(
    rewards,
    group_size: int,
    advantage_eps: float
):
    """
    Args:
        rewards: List[float]: rewards computed by compare sampled result with groundtruth
        group_size (int): Number of responses per question (group).
        advantage_eps (float): make the ground true not equal to 0

    Returns:
        advatanges: List[float], same shape with rewards
    """
    rollout_batch_size = len(rewards)
    assert (
        rollout_batch_size % group_size == 0
    ), f"rollout_batch_size %% group_size = {rollout_batch_size % group_size}"

    n_prompts = rollout_batch_size // group_size
    advantages = np.empty(shape=(rollout_batch_size,))

    rewards = np.array(rewards)
    for i in range(n_prompts):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size   

        rewards_g = rewards[start_idx:end_idx]
        mu = np.mean(rewards_g)
        std = np.sqrt(
            np.mean((rewards_g - mu) * (rewards_g - mu))
            * (group_size / (group_size - 1))
        ) 
        advantages[start_idx:end_idx] = (rewards_g - mu) / (std + advantage_eps)

    advantages = list(advantages)
    return advantages

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """policy-gradient loss at every token (advantages adjusted)

    loss = -log(o_i | o_{0..i-1}) * adv

    Args:
        raw_rewards_or_advantages (torch.Tensor): Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs (torch.Tensor): torch.Tensor Shape (batch_size, sequence_length), logprobs for each token.

    Returns:
        torch.Tensor: Shape (batch_size, sequence_length), the per-token policy-gradient loss
    """
    return -policy_log_probs * raw_rewards_or_advantages


def compute_grpo_clip_loss(
    advantages: torch.Tensor | float,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """computes the per-token GRPO-Clip loss

    -min(
        pi_new / pi_old * advantage,
        clip(pi_new / pi_old) * advantage
    )

    Args:
        advantages (torch.Tensor): Shape (batch_size, 1), per-example advantages A.
        policy_log_probs (torch.Tensor): Shape (batch_size, sequence_length), per-token log probs from the policy being trained.
        old_log_probs (torch.Tensor): Shape (batch_size, sequence_length), per-token log probs from the old policy.
        cliprange (float): float Clip parameter ϵ (e.g. 0.2).

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            loss; torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
            info: other recording infos
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    ratio_clip = torch.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # static: reach bar
    with torch.no_grad():
        hit_mask = (ratio > 1.0 + cliprange) | (ratio < 1.0 - cliprange)
        num_hit = torch.sum(hit_mask)
        hit_bar_ratio = num_hit / len(ratio)

    loss = -torch.minimum(ratio * advantages, ratio_clip * advantages)
    return loss, {'hit_bar_ratio': hit_bar_ratio}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Select and compute the desired policy-gradient loss.

    Args:
        policy_log_probs (torch.Tensor): (batch_size, sequence_length) per-token log-probabilities
        loss_type (Literal[&quot;no_baseline&quot;, &quot;reinforce_with_baseline&quot;, &quot;grpo_clip&quot;])
        raw_rewards (torch.Tensor | None, optional): Required if loss_type == "no_baseline"; shape (batch_size, 1). Defaults to None.
        advantages (torch.Tensor | None, optional): Required for "reinforce_with_baseline" and "grpo_clip"; 
            shape (batch_size, 1). Defaults to None.
        old_log_probs (torch.Tensor | None, optional): Required for "grpo_clip"; shape (batch_size, sequence_length). Defaults to None.
        cliprange (float | None, optional): Required for "grpo_clip"; scalar ϵ used for clipping. Defaults to None.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss (batch_size, sequence_length), per-token loss.
            metadata dict, statistics from the underlying routine
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}

    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None

        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    else:
        raise NotImplementedError


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.

    Args:
        tensor (torch.Tensor): The data to be averaged
        mask (torch.Tensor): Same shape as tensor; positions with 1 are included in the mean.
        dim (int | None, optional): Dimension over which to average. If None, compute the mean over all masked elements.. Defaults to None.

    Returns:
        torch.Tensor
    """
    assert tensor.shape == mask.shape

    dim_sum = torch.sum(mask, dim=dim)
    if not torch.all(dim_sum != 0):
        warnings.warn("all dimension in some slice is 0, NaN raise in result")
    # mask_dim_sum_equal_0 = dim_sum == 0

    average = torch.sum(tensor * mask, dim=dim) / dim_sum
    # average = torch.where(mask_dim_sum_equal_0, 0.0, average)
    # assert average.dtype == tensor.dtype

    return average


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs (torch.Tensor): (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
        response_mask (torch.Tensor): (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps (int): Number of microbatches per optimizer step.
        loss_type (Literal[&quot;no_baseline&quot;, &quot;reinforce_with_baseline&quot;, &quot;grpo_clip&quot;]): _description_
        raw_rewards (torch.Tensor | None, optional): Needed when loss_type == "no_baseline"; shape (batch_size, 1). Defaults to None.
        advantages (torch.Tensor | None, optional): Needed when loss_type != "no_baseline"; shape (batch_size, 1). Defaults to None.
        old_log_probs (torch.Tensor | None, optional): Required for GRPO-Clip; shape (batch_size, sequence_length). Defaults to None.
        cliprange (float | None, optional): Clip parameter ϵ for GRPO-Clip. Defaults to None.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return  this so we can log it.

        metadata Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.

        call loss.backward() in this function. Make sure to adjust for gradient accumulation.
    """
    # compute loss at each token
    losses, _ = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    
    # micro_batch = losses.shape[0]
    # loss = (losses * response_mask).sum()
    # losses /= gradient_accumulation_steps
    loss = masked_mean(losses, response_mask, dim=1) # now loss of shape [bsize,]
    loss = loss.mean()
    loss /= gradient_accumulation_steps

    loss.backward()

    return loss, {}