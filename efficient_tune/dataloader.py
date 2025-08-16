from tqdm import tqdm
import torch
import numpy as np
from efficient_tune.trainer.util_sft import tokenize_prompt_and_output
from efficient_tune.evaluate import generate_answer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from efficient_tune.drgrpo_grader import r1_zero_reward_fn
from efficient_tune.trainer.util_grpo import compute_advantages


def apply_r1_template_to_problem(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


def apply_r1_template_to_solution(solution: str, answer: str):
    return solution + " </think> <answer> " + answer + " </answer>"


def make_train_dataloader(
    train_data_path, tokenizer, batch_size=5, shuffle=True, max_train_tokens=1024
):
    dataset = load_from_disk(train_data_path)

    data = dataset.map(
        lambda row: {
            "prompt": apply_r1_template_to_problem(row["problem"]),
            "response": apply_r1_template_to_solution(row["solution"], row["answer"]),
        }
    )

    print(f"train dataset contains {len(dataset)} items")

    class CollateFn:
        def __init__(self, tokenizer, max_train_tokens=1024):
            self.tokenizer = tokenizer
            self.max_tokens = max_train_tokens

        def __call__(self, batch):
            """

            Args:
                batch : dictionary contains prompt and responses

            Returns:
                dict: contains
                    "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
                        the tokenized prompt and output strings, with the final token sliced off
                    "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
                        shifted input ids, i.e., the input ids without the first token.
                    "response_mask" torch.Tensor of shape (batch_size, max(prompt_and_output_lens)
            """
            # batch = [{'problem'...},{...},...]
            prompts = [row["prompt"] for row in batch]
            responses = [row["response"] for row in batch]

            prompt_and_label = tokenize_prompt_and_output(
                prompts, responses, self.tokenizer, shift_label=False
            )

            seqlen = prompt_and_label["input_ids"].shape[1]
            if seqlen > self.max_tokens:
                # sampling start from [0, seqlen - max_train_tokens]
                start_idx = np.random.randint(0, seqlen - self.max_tokens + 1)
                end_pos = start_idx + self.max_tokens
                for key in prompt_and_label:
                    prompt_and_label[key] = prompt_and_label[key][:, start_idx:end_pos]

            return prompt_and_label

    collate_fn = CollateFn(tokenizer, max_train_tokens)
    dataloader = DataLoader(
        dataset=data, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
    )

    return dataloader


def make_eval_dataloader(eval_data_path, batch_size=5, shuffle=True):
    dataset = load_from_disk(eval_data_path)

    dataset = dataset.map(
        lambda row: {"prompt": apply_r1_template_to_problem(row["problem"])}
    )

    def collate_fn(batch):
        prompts = [row["prompt"] for row in batch]
        targets = [row["answer"] for row in batch]
        return {"prompt": prompts, "answer": targets}

    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
    )

    return dataloader


def make_grpo_dataloader(
    train_data_path,
    model,
    tokenizer,
    num_prompts,
    group_size,
    advantage_eps=1e-8,
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=1024,
    shuffle=True,
):
    eval_configs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "output_scores": False,
        "output_logits": True,
    }

    dataset = load_from_disk(train_data_path)
    dataset = dataset.map(
        lambda row: {"prompt": apply_r1_template_to_problem(row["problem"])}
    )

    class CollateFn:
        def __init__(
            self, model, tokenizer, group_size: int, advantage_eps: float, eval_configs
        ):
            self.tokenizer = tokenizer
            self.group_size = group_size
            self.model = model
            self.advantage_eps = advantage_eps
            self.eval_configs = eval_configs

        def __call__(self, batch):
            train_data = {
                "prompt": [],               # prompts of current problem list[str]
                "groudtruth": [],           # answsers of problems: list[str]
                "generated_text": [],       # generated texts by the model, exclude prompts: list[str]
                "generated_tokens": [],     # correspond tokens: list[tensor[int]]
                "sequences": [],            # all tokens including prompts: list[tensor[int]]
                "prompt_token_len": [],     # prefix length
                "generated_token_len": [],  # len of tokens exclude prompt
                "generated_logsmx": [],     # log prob(token) for token in generated_tokens: list[tensor[float32]]
                "format_reward": [],
                "answer_reward": [],
                "reward": [],               # reward of each anwser list[float]
                "advantages": []            # adv of each anwser list[float] if a question too hard, all adv can be 0.0 due to reward == 0.0
            }

            # change global variable in model (change it back in train)
            self.model.eval()
            for row in tqdm(batch, desc="generate grpo batch"):
                for _ in range(self.group_size):
                    prompt = row["prompt"]
                    groudtruth = row["answer"]

                    output_text, model_outputs = generate_answer(
                        self.model, self.tokenizer, [prompt], **eval_configs
                    )

                    output_text = output_text[0]
                    generated_tokens = model_outputs["generated_tokens"][0]
                    output_score = torch.concat(
                        model_outputs["logits"]
                    )  # [seqlen, num_vocab]
                    output_score = torch.log_softmax(output_score, dim=1)
                    output_logsmx = torch.gather(
                        output_score, dim=1, index=generated_tokens.unsqueeze(dim=1)
                    ).squeeze(dim=1)

                    reward_info = r1_zero_reward_fn(output_text, groudtruth, fast=False)

                    train_data["prompt"].append(prompt)
                    train_data["groudtruth"].append(groudtruth)
                    train_data["generated_text"].append(output_text)
                    train_data["generated_tokens"].append(generated_tokens.to("cpu"))
                    train_data["generated_logsmx"].append(output_logsmx.to("cpu"))
                    train_data["sequences"].append(
                        model_outputs["sequences"][0].to("cpu")
                    )

                    sequence_len = len(train_data["sequences"][-1])
                    generated_token_len = len(train_data["generated_tokens"][-1])
                    assert sequence_len >= generated_token_len
                    train_data["generated_token_len"].append(generated_token_len)
                    train_data["prompt_token_len"].append(
                        sequence_len - generated_token_len
                    )

                    for key in reward_info:
                        train_data[key].append(reward_info[key])

            train_data["advantages"] = compute_advantages(
                train_data["reward"], self.group_size, self.advantage_eps
            )

            # assert len(train_data["prompt"]) == self.group_size * len(batch)
            self.verify(train_data)
            return train_data
        
        def verify(self, batch):
            batch_size = len(batch['generated_text'])
            assert batch_size % self.group_size == 0
            n_prompts = batch_size // self.group_size

            
            for i in range(n_prompts):
                start_idx = i * self.group_size
                end_idx = (i+1) * self.group_size
                expect_prompt_len = batch['prompt_token_len'][start_idx]
                expect_prompt_tokens = batch['sequences'][start_idx][:expect_prompt_len]
                for j in range(start_idx, end_idx):
                    # for same group the prompt_token_len shall equal
                    assert batch['prompt_token_len'][j] == expect_prompt_len
                    assert torch.all(
                        batch['sequences'][j][:expect_prompt_len] == expect_prompt_tokens
                    )

                    # generated_logsmx shall have the same length with generated_tokens
                    assert batch['generated_logsmx'][j].shape == (batch['generated_token_len'][j],)


    collate_fn = CollateFn(model, tokenizer, group_size, advantage_eps, eval_configs)
    dataloader = DataLoader(
        dataset=dataset, batch_size=num_prompts, collate_fn=collate_fn, shuffle=shuffle
    )
    return dataloader
