from efficient_tune.trainer.util_sft import sft_microbatch_train_step_using_network
from efficient_tune.evaluate import evaluate_step
import os
import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from efficient_tune.dataloader import make_train_dataloader, make_eval_dataloader
from efficient_tune.tensorboard_logger import make_tensorboard_logger
from .optimizer import make_optimizer, make_scheduler


SFT_TRAINER_DEFAULT_CONFIG = {
    "learning_rate": 5e-5,
    "learning_rate_min": 1e-6,
    "betas": (0.9, 0.999),
    "weight_decay": 1e-3,
    "eps": 1e-8,
    "warmup_steps": 100,
    "max_steps": 1000, # step for perform optimizer.step
    "batch_size": 1,
    "gradient_accumulation_steps": 32,
    "max_grad_norm": 1.0,  # Gradient clipping
    "save_steps": 200,
    "save_dir": "data/output/saved_model",
    "log_dir": "data/output/log",
    "eval_steps": 20,
    "logging_steps": 10,
    "max_train_tokens": 1024,
    "train_dataset_path": "./data/math/train",
    # evaluate configs
    "eval_dataset_path": "./data/math/eval",
    "eval_batch_size": 1,
    "eval_max_iteration": 40, # samples used to eval: 1 * 40 = 40
    "eval_temperature": True,
    "eval_top_p": 1.0,
    "eval_max_new_tokens": 1024
}

class STFTrainer:
    def __init__(self, model, tokenizer, trainer_config):
        self.model = model
        self.tokenizer = tokenizer

        self.dataloader_train = make_train_dataloader(trainer_config['train_dataset_path'], tokenizer, batch_size=trainer_config['batch_size'], max_train_tokens=trainer_config['max_train_tokens'])
        self.dataloader_eval = make_eval_dataloader(trainer_config['eval_dataset_path'], batch_size=trainer_config['eval_batch_size'])
        self.optimizer = make_optimizer(model.parameters(), trainer_config)
        self.scheduler = make_scheduler(self.optimizer, trainer_config)

        self.trainer_config = trainer_config
        self.iter_eval = iter(self.dataloader_eval)
        self.iter_train = iter(self.dataloader_train)
        self.logger = make_tensorboard_logger(trainer_config, "STFTrainer")

    def train(self):
        # self.model.train()
        # gradient_accumulation_steps = trainer_config['gradient_accumulation_steps']
        # iterator = iter(self.dataloader_train)

        # self.save_model(0)
        for i in tqdm(range(self.trainer_config['max_steps']), desc='train'):
            infos = self._train_step()
            # print(f'iter={i}, infos={infos}')
            self.logger.log_metrics(infos, step=i)

            if (i+1) % self.trainer_config['eval_steps'] == 0:
                eval_info = self.evaluate()
                # print(f'iter={i}, eval_infos={eval_info}')
                self.logger.log_metrics(eval_info, step=i)

            # check point model
            if (i+1) % self.trainer_config['save_steps'] == 0:
                self.save_model(i+1)
        
        self.save_model(self.trainer_config['max_steps'])
        return self.model
    
    def save_model(self, iteration):
        save_dir = self.trainer_config['save_dir']
        save_dir = os.path.join(save_dir, f'iteration={iteration}')
        print(f"save model to {save_dir}")

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.model.save_pretrained(save_dir)
    
    def _train_step(self):
        self.model.train()
        gradient_accumulation_steps = self.trainer_config['gradient_accumulation_steps']
        loss = 0.0
        for i in range(gradient_accumulation_steps):
            # get data batch
            try:
                batch = next(self.iter_train)

            except StopIteration:
                self.iter_train = iter(self.dataloader_train)
                batch = next(self.iter_train)
            

            for key in batch:
                batch[key] = batch[key].to(self.model.device)

            # get model ouput
            # model_output = get_response_log_probs(model, batch['input_ids'], batch['labels'])

            # get loss
            # loss, _ = sft_microbatch_train_step(
            #     model_output['log_probs'],
            #     batch['response_mask'],
            #     gradient_accumulation_steps,
            #     normalize_constant
            # )
            try:
                loss_micro = sft_microbatch_train_step_using_network(self.model, batch, gradient_accumulation_steps)
                loss += loss_micro.item()

            except torch.OutOfMemoryError:
                print(f"shape of input tensor: {batch['input_ids'].shape}" )
                raise torch.OutOfMemoryError
                
        # apply gradient
        #if (i+1) % gradient_accumulation_steps == 0:
        grad_norm_old = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.trainer_config['max_grad_norm']
        )
        grad_norm_old = grad_norm_old.item()

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # report 
        infos = {
            'loss': loss,
            'grad_norm': grad_norm_old,
            'lr': self.scheduler.get_last_lr()[0]
        }

        return infos
    
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
            try:
                batch = next(self.iter_eval)

            except StopIteration:
                self.iter_eval = iter(self.dataloader_eval)
                batch = next(self.iter_eval)

            self._eval_step(batch, evaluate_infos)

        evaluate_infos_agg = {}
        for key in evaluate_infos:
            evaluate_infos_agg[key] = np.mean(np.array(evaluate_infos[key]))

        return evaluate_infos_agg
