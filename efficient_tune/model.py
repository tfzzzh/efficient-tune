import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BitsAndBytesConfig 
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from peft.tuners.lora import LoraLayer




######## 
# DEFAULT CONFIG
#########
MODEL_DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen2.5-Math-1.5B",
    "torch_dtype": "bfloat16", 
    "attn_implementation": "flash_attention_2",
    "lora_r": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "bnb_compute_type": 'bfloat16',
    "bnb_load_in_4bit": True,
    "bnb_load_in_8bit": False,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4"
}


def make_model_and_tokenizer(model_config):
    pretrain_config = {}
    pretrain_config['torch_dtype'] = model_config['torch_dtype']
    pretrain_config['attn_implementation'] = model_config['attn_implementation']
    pretrain_config['device_map'] = 'auto'
    # pretrain_config['max_memory'] = {0:'8000MB'}
    pretrain_config['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=model_config['bnb_load_in_4bit'],
        load_in_8bit=model_config['bnb_load_in_8bit'],
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=model_config['bnb_compute_type'],
        bnb_4bit_use_double_quant=model_config['bnb_4bit_use_double_quant'],
        bnb_4bit_quant_type=model_config['bnb_4bit_quant_type']
    )

    model_name = model_config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, **pretrain_config
    )
    # model = model.to(device)

    model = apply_lora(model, model_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    return model, tokenizer


def apply_lora(model, model_config):
    modules = find_all_linear_names(
        model, load_in_4bit=model_config['bnb_load_in_4bit'], 
        load_in_8bit=model_config['bnb_load_in_8bit']
    )

    lora_config = LoraConfig(
        r=model_config['lora_r'],
        lora_alpha=model_config['lora_alpha'],
        target_modules=modules,
        lora_dropout=model_config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config)
    print("applying lora to model complete")

    # patch data type
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if model_config['bnb_compute_type'] == "bfloat16":
                # print('dtype of lora layer is ', list(module.parameters())[0].dtype)
                module = module.to(torch.bfloat16)

        if "norm" in name:
            module = module.to(torch.float32) # bloat32
            module = make_rmsnorm_mixed(module)

        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if (
                    model_config['bnb_compute_type'] == "bfloat16"
                ):
                    module = module.to(torch.bfloat16)

    return model


def find_all_linear_names(model, load_in_4bit=True, load_in_8bit=False):
    import bitsandbytes as bnb

    if load_in_4bit:
        layer_cls = bnb.nn.Linear4bit

    elif load_in_8bit:
        layer_cls = bnb.nn.Linear8bitLt

    else:
        layer_cls = torch.nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, layer_cls):
            names = name.split(".")
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        # print('remove lm_head')
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


# Utility to wrap rmsnorm to 32 bit
def make_rmsnorm_mixed(m):
    orig_forward = m.forward
    def mixed_forward(x: torch.Tensor):
        # 1) cast to fp32 for stable RMS computation
        x_fp32 = x.to(torch.float32)
        y_fp32 = orig_forward(x_fp32)
        # 2) cast result back to the dtype of the rest of the model
        return y_fp32.to(x.dtype)
    m.forward = mixed_forward
    return m