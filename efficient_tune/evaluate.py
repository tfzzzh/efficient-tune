from efficient_tune.drgrpo_grader import r1_zero_reward_fn
import torch


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompts,
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=1024,
    output_scores=False,
    output_logits=False,
):
    # Configure generation parameters to mimic vLLM
    generation_config = {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": output_scores,
        "output_logits": output_logits,
        "stop_strings": ["</answer>"],
    }

    inputs = tokenizer(
        prompts, padding_side="left", return_tensors="pt", padding=True
    ).to(model.device)
    model_outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        tokenizer=tokenizer,
        **generation_config,
    )

    generated_tokens = model_outputs.sequences[:, inputs["input_ids"].shape[1] :]
    model_outputs['generated_tokens'] = generated_tokens
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return generated_text, model_outputs


def evaluate_step(
    model, tokenizer, batch, temperature, top_p, max_new_tokens, infos_out
):
    model.eval()
    prompts = batch["prompt"]
    targets = batch["answer"]
    preds, _ = generate_answer(
        model,
        tokenizer,
        prompts,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    for pred, groudtruth in zip(preds, targets):
        info = r1_zero_reward_fn(pred, groudtruth, fast=False)
        for key, value in info.items():
            infos_out[key].append(value)
