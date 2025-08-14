import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def make_optimizer(parameters, optimizer_config):
    opt = optim.AdamW(
        parameters,
        lr=optimizer_config["learning_rate"],
        betas=optimizer_config["betas"],
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    )
    return opt

def make_scheduler(optimizer, config):
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config['max_steps'],  # Adjust based on your training steps
        eta_min=config['learning_rate_min']# Minimum learning rate
    )
    return scheduler
