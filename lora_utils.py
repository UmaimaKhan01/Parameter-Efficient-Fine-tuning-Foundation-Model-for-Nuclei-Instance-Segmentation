# Save this code as lora_utils.py in your working directory

import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

def inject_lora(image_encoder):
    """
    Injects LoRA into the image encoder of MobileSAM
    """
    lora_config = LoraConfig(
        r=8,               # rank
        lora_alpha=32,     # scaling factor
        target_modules=["qkv"],  # modify only the attention qkv projection layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    
    lora_model = get_peft_model(image_encoder, lora_config)
    print("✅ LoRA successfully injected into MobileSAM encoder.")
    return lora_model

def count_lora_parameters(model):
    """
    Count trainable LoRA parameters only
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
