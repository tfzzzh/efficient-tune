import argparse
from efficient_tune.model import make_model_and_tokenizer, MODEL_DEFAULT_CONFIG
from efficient_tune.trainer import STFTrainer, GRPOTrainer, SFT_TRAINER_DEFAULT_CONFIG, GRPO_TRAINER_DEFAULT_CONFIG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=str, default='grpo')
    args = parser.parse_args()
    model, tokenizer = make_model_and_tokenizer(MODEL_DEFAULT_CONFIG)

    if args.trainer == 'sft':
        trainer = STFTrainer(model, tokenizer, SFT_TRAINER_DEFAULT_CONFIG)
        trainer.train()

    elif args.trainer == 'grpo':
        trainer = GRPOTrainer(model, tokenizer, GRPO_TRAINER_DEFAULT_CONFIG)
        trainer.train()

    else:
        raise NotImplementedError(f"Trainer {args.trainer} is not supported. Currently only sft and grpo trainers are supported")

if __name__ == '__main__':

    main()