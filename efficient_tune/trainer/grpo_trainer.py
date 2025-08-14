import torch
from torch import nn
import numpy as np
import os


from efficient_tune.evaluate import evaluate_step
from efficient_tune.dataloader import make_grpo_dataloader, make_eval_dataloader
from tqdm import tqdm
from efficient_tune.trainer.grpo import compute_grpo_clip_loss
from efficient_tune.tensorboard_logger import (
    make_tensorboard_logger,
)
from .optimizer import (
    make_optimizer,
    make_scheduler,
)

GRPO_TRAINER_DEFAULT_CONFIG = {
    "cliprange": 0.2,
    # config for adam
    "learning_rate": 5e-5,
    "learning_rate_min": 1e-5,
    "betas": (0.9, 0.999),
    "weight_decay": 0.0,
    "eps": 1e-8,
    # config for Scheduler
    "warmup_steps": 100,
    "max_steps": 1000,      # step for perform optimizer.step (max_steps = n_grpo_epoch_per_batch * num_batch_sampled)
    # step and batch_size
    "num_prompts": 4,               # how many problem will be sampled in a train batch
    "group_size_per_prompts": 8,    # how many answer will be sampled per question (batch_size = num_prompt * group_size_per_prompts)
    "train_max_new_tokens": 800,
    "n_grpo_epoch_per_batch": 10,  # how many update shall I run when collect a training batch
    # "max_number_batch": 1,    # max_steps // n_grpo_steps
    "max_grad_norm": 1.0,  # Gradient clipping
    "save_steps": 10,
    "save_dir": "data/output/saved_model/grpo",
    "log_dir": "data/output/log",
    "eval_steps": 4,
    "logging_steps": 10,
    "train_dataset_path": "./data/math/train",
    # evaluate configs
    "eval_dataset_path": "./data/math/eval",
    "eval_batch_size": 1,
    "eval_max_iteration": 40,  # samples used to eval: 1 * 40 = 40
    "eval_temperature": True,
    "eval_top_p": 1.0,
    "eval_max_new_tokens": 1024,
}


class GRPOTrainer:
    def __init__(self, model, tokenizer, trainer_config):
        self.model = model
        self.tokenizer = tokenizer

        self.dataloader_train = make_grpo_dataloader(
            trainer_config["train_dataset_path"],
            model,
            tokenizer,
            trainer_config["num_prompts"],
            trainer_config["group_size_per_prompts"],
            max_new_tokens=trainer_config["train_max_new_tokens"],
        )
        self.dataloader_eval = make_eval_dataloader(
            trainer_config["eval_dataset_path"],
            batch_size=trainer_config["eval_batch_size"],
        )
        self.optimizer = make_optimizer(model.parameters(), trainer_config)
        self.scheduler = make_scheduler(self.optimizer, trainer_config)

        self.trainer_config = trainer_config
        self.iter_eval = iter(self.dataloader_eval)
        self.iter_train = iter(self.dataloader_train)
        self.logger = make_tensorboard_logger(trainer_config, "GRPOTrainer")

        self.device = self.model.device
        self.train_dtype = model.config.torch_dtype

    def train(self):
        num_loop = (
            self.trainer_config["max_steps"]
            // self.trainer_config["n_grpo_epoch_per_batch"]
        )
        for i in tqdm(range(num_loop), desc="train"):
            train_info = self._train_step()
            self.logger.log_metrics(train_info, step=i)

            # check point model
            if (i+1) % self.trainer_config['save_steps'] == 0:
                self.save_model(i+1)

            if (i+1) % self.trainer_config['eval_steps'] == 0:
                eval_info = self.evaluate()
                # print(f'iter={i}, eval_infos={eval_info}')
                self.logger.log_metrics(eval_info, step=i)

    def save_model(self, iteration):
        save_dir = self.trainer_config['save_dir']
        save_dir = os.path.join(save_dir, f'iteration={iteration}')
        print(f"save model to {save_dir}")

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.model.save_pretrained(save_dir)

    def _train_step(self):
        self.model.train()
        batch = self._get_train_batch()
        infos = {}
        for _ in range(self.trainer_config["n_grpo_epoch_per_batch"]):
            info = self._update(batch)

            for key in info:
                if key not in infos:
                    infos[key] = []
                infos[key].append(info[key])

        # aggregate infos
        infos = {key: np.mean(np.array(infos[key])) for key in infos}

        # compute train reward for this batch (lagged reward)
        infos['train_reward'] = np.mean(np.array(batch['reward']))
        infos['train_format_reward'] = np.mean(np.array(batch['format_reward']))
        infos['train_output_length'] = np.mean(np.array(batch['generated_token_len']))

        return infos

    def _update(self, batch):
        infos = {}  # update infos
        batch_size = len(batch["sequences"])

        # aggregate gradient over the batch
        loss_batch = 0.0
        hit_bar_ratio = 0.0
        for i in range(batch_size):
            # get data and move to cuda
            (
                sequences,
                generated_tokens,
                generated_logsmx,
                advantage,
                prompt_token_length,
            ) = self._prepare_train_data(batch, i)

            # get model logits (I should combine with prompt tokens)
            input_ids = sequences[:-1].unsqueeze(0)
            logits = self.model(
                input_ids=input_ids,
            ).logits
            logits = logits.squeeze(0)  # [seqlen, num_vocab]

            logits = logits[prompt_token_length - 1 :, :]
            logprob = torch.log_softmax(logits, dim=1)  # [seqlen, num_vocab]
            logprob = torch.gather(logprob, dim=1, index=generated_tokens.unsqueeze(1))
            logprob = logprob.squeeze(dim=1)
            assert logprob.shape == generated_logsmx.shape

            # compute loss
            loss, loss_info = compute_grpo_clip_loss(
                advantage, logprob, generated_logsmx, self.trainer_config['cliprange']
            )
            loss = loss.mean()
            loss /= batch_size

            # compute gradient
            loss.backward()

            # record loss
            loss_batch += loss.item()
            hit_bar_ratio += loss_info['hit_bar_ratio'].item() / batch_size      

        # normalize gradient
        grad_norm_old = nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.trainer_config["max_grad_norm"]
        )
        grad_norm_old = grad_norm_old.item()

        # apply optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        infos['loss'] = loss_batch
        infos['grad_norm'] = grad_norm_old
        infos['hit_bar_ratio'] = hit_bar_ratio
        return infos

    def _prepare_train_data(self, batch, idx):
        sequences = batch["sequences"][idx]
        generated_tokens = batch["generated_tokens"][idx]
        generated_logsmx = batch["generated_logsmx"][idx]
        advantage = batch["advantages"][idx]
        prompt_token_length = batch["prompt_token_len"][idx]
        assert prompt_token_length > 0

        sequences = sequences.to(self.device)
        generated_tokens = generated_tokens.to(self.device)
        generated_logsmx = generated_logsmx.to(self.device).to(self.train_dtype)

        assert sequences.dtype == torch.long
        assert generated_tokens.dtype == torch.long
        assert generated_logsmx.dtype == self.train_dtype

        assert sequences.shape == (generated_tokens.shape[0] + prompt_token_length,)
        assert generated_logsmx.shape == generated_tokens.shape
        assert isinstance(advantage, float)

        return (
            sequences,
            generated_tokens,
            generated_logsmx,
            advantage,
            prompt_token_length,
        )

    def _get_train_batch(self):
        """get one databatch from the dataset"""
        try:
            batch = next(self.iter_train)

        except StopIteration:
            self.iter_train = iter(self.dataloader_train)
            batch = next(self.iter_train)

        return batch
    
    def _get_eval_batch(self):
        try:
            batch = next(self.iter_eval)

        except StopIteration:
            self.iter_eval = iter(self.dataloader_eval)
            batch = next(self.iter_eval)
        return batch


    def _eval_step(
        self, batch, infos_out
    ):
        evaluate_step(
            self.model, 
            self.tokenizer, 
            batch, 
            self.trainer_config['eval_temperature'],
            self.trainer_config['eval_top_p'],
            self.trainer_config['eval_max_new_tokens'],
            infos_out
        )

    def evaluate(self):
        max_iteration = self.trainer_config['eval_max_iteration']
        evaluate_infos = {
            'format_reward': [],
            'answer_reward': [],
            'reward': []
        }

        for i in tqdm(range(max_iteration), desc="evaluate"):
            batch = self._get_eval_batch()

            self._eval_step(batch, evaluate_infos)

        evaluate_infos_agg = {}
        for key in evaluate_infos:
            evaluate_infos_agg[key] = np.mean(np.array(evaluate_infos[key]))

        return evaluate_infos_agg