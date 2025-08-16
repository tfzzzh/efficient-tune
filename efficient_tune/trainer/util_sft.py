from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import torch


def tokenize_prompt_and_output(
    prompt_strs: List[str], output_strs: List[str], tokenizer: PreTrainedTokenizer,
    shift_label: bool = True
):
    """Tokenize the prompt and output strings, and construct a mask

    Args:
        prompt_strs (List[str]): List of prompt strings.
        output_strs (List[str]): List of output strings.
        tokenizer (PreTrainedTokenizer): Tokenizer to use for tokenization.
        shift_label (bool): when label is not shifted the loss is given by the neural net itself

    returns:
        when shift_label is true
            dict[str, torch.Tensor]:
                "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
                    the tokenized prompt and output strings, with the final token sliced off
                "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
                    shifted input ids, i.e., the input ids without the first token.
                "response_mask" torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
                1): a mask on the response tokens in the labels.
        otherwise
            dict[str, torch.Tensor]:
                "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens))
                    the tokenized prompt and output strings
                "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens))
                "attention_mask" torch.Tensor of shape (batch_size, max(prompt_and_output_lens))
    """
    assert len(prompt_strs) == len(
        output_strs
    ), "prompt and output shall of the same length"
    n = len(prompt_strs)

    prompt_tokens = tokenizer(prompt_strs, padding=False, add_special_tokens=False)[
        "input_ids"
    ]
    output_tokens = tokenizer(output_strs, padding=False, add_special_tokens=False)[
        "input_ids"
    ]

    batch_length = max(
        len(ptoken) + len(otoken)
        for (ptoken, otoken) in zip(prompt_tokens, output_tokens)
    )

    data_tokens = torch.empty(n, batch_length, dtype=torch.long)
    response_mask = torch.zeros(n, batch_length, dtype=torch.bool)
    data_tokens[:] = tokenizer.pad_token_id

    for i in range(n):
        ptoken = prompt_tokens[i]
        otoken = output_tokens[i]

        len_p = len(ptoken)
        len_o = len(otoken)

        data_tokens[i, :len_p] = torch.tensor(ptoken, dtype=torch.long)
        data_tokens[i, len_p : len_p + len_o] = torch.tensor(otoken, dtype=torch.long)
        response_mask[i, len_p : len_p + len_o] = True

    if shift_label:
        result = {
            "input_ids": data_tokens[:, :-1],
            "labels": data_tokens[:, 1:],
            "response_mask": response_mask[:, 1:],
        }
    else:
        result = {
            "input_ids": data_tokens,
            "labels": data_tokens,
            "attention_mask": response_mask
        }

    return result


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): Tensor of shape (batch_size, sequence_length, vocab_size)

    Returns:
        torch.Tensor: Shape (batch_size, sequence_length)
    """
    # compute stable logits which remove max of last dim
    logits_stable = logits - logits.max(dim=-1, keepdim=True).values

    # compute log exp(logits) / sum exp(logits)
    # which equals: logits - log sum exp(logits)
    exp = torch.exp(logits_stable)
    normalizer = torch.sum(exp, dim=-1, keepdim=True)
    smx = exp / normalizer
    log_smx = logits_stable - torch.log(normalizer)

    entropy = -(smx * log_smx).sum(dim=-1)

    return entropy


# def get_response_log_probs(
#     model: PreTrainedModel,
#     input_ids: torch.Tensor,
#     labels: torch.Tensor,
#     return_token_entropy: bool = False,
# ) -> dict[str, torch.Tensor]:
#     """

#     Args:
#         model (PreTrainedModel): HuggingFace model used for scoring
#         input_ids (torch.Tensor): shape (batch_size, sequence_length)
#         labels (torch.Tensor): shape (batch_size, sequence_length)
#         return_token_entropy (bool, optional): If True, also return per-token entropy

#     Returns:
#         dict[str, torch.Tensor]:
#             "log_probs" shape (batch_size, sequence_length)
#             "token_entropy" optional, shape (batch_size, sequence_length)

#     """
#     # Obtain logits with model(input_ids).logits
#     logits = model(input_ids).logits

#     # compute stable logits which remove max of last dim
#     logits_stable = logits - logits.max(dim=-1, keepdim=True).values

#     # compute log exp(logits) / sum exp(logits)
#     # which equals: logits - log sum exp(logits)
#     exp = torch.exp(logits_stable)
#     normalizer = torch.sum(exp, dim=-1, keepdim=True)
#     log_smx = logits_stable - torch.log(normalizer)

#     # slice out log probs using gather
#     log_probs = torch.gather(log_smx, dim=-1, index=labels.unsqueeze(-1))
#     log_probs = log_probs.squeeze(-1)

#     result = {"log_probs": log_probs}
#     if return_token_entropy:
#         smx = exp / normalizer
#         result["token_entropy"] = -(smx * log_smx).sum(dim=-1)

#     return result


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """

    Args:
        model (PreTrainedModel): HuggingFace model used for scoring
        input_ids (torch.Tensor): shape (batch_size, sequence_length)
        labels (torch.Tensor): shape (batch_size, sequence_length)
        return_token_entropy (bool, optional): If True, also return per-token entropy

    Returns:
        dict[str, torch.Tensor]:
            "log_probs" shape (batch_size, sequence_length)
            "token_entropy" optional, shape (batch_size, sequence_length)

    """
    # Obtain logits with model(input_ids).logits
    logits = model(input_ids).logits
    logps = selective_log_softmax(logits, labels) 
    result = {"log_probs": logps}

    if return_token_entropy:
        smx = torch.softmax(logits, dim=-1)
        result["token_entropy"] = -(smx * torch.log(smx)).sum(dim=-1)

    return result


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    A memory efficient implementation of `log_softmax` -> `gather` operation
    equivalent to
    ```python
    logps = torch.gather(logits.softmax(-1), dim=-1 index=index.unsqueeze(-1)).squeeze(-1)
    ```
    
    Args:
        logits (torch.Tensor): tensor of shape (..., num_classes)
        index (torch.Tensor): shape (...), specifying the selected classes

    Returns:
        torch.Tensor: Gathered log probabilities with the same shape with index
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(dim=-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        # logsumexp approach is unstable with bfloat16,
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logp = torch.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logp.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    
    return per_token_logps

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """

    Args:
        tensor (torch.Tensor): The tensor to sum and normalize.
        mask (torch.Tensor): Same shape as tensor; positions with 1 are included in the sum
        normalize_constant (float): the constant to divide by for normalization.
        dim (int | None, optional): the dimension to sum along before normalization

    Returns:
        torch.Tensor: the normalized sum
    """
    return (tensor * mask).sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch

    Args:
        policy_log_probs (torch.Tensor): (batch_size, sequence_length), per-token log-probabilities from the SFT policy being trained.
        response_mask (torch.Tensor): (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding
        gradient_accumulation_steps (int): Number of microbatches per optimizer step
        normalize_constant (float, optional): The constant by which to divide the sum. It is fine to leave this as 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss scalar tensor.
            metadata: data you wish to log

    call loss.backward() in this function. Make sure to adjust for gradient accumulation.
    """
    loss = masked_normalize(policy_log_probs, response_mask, normalize_constant, -1)
    loss = -(loss.sum() / gradient_accumulation_steps)

    loss.backward()

    return loss, {'loss': loss}


def sft_microbatch_train_step_using_network(
    model: PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
    gradient_accumulation_steps: int
) -> torch.Tensor:
    loss = model(**inputs).loss
    loss /= float(gradient_accumulation_steps)
    loss.backward()

    return loss
