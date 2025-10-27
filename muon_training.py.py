import os
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers

from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    Trainer
)
from loguru import logger
import datasets
transformers.logging.set_verbosity_error()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

import torch
from torch import nn
from torch.optim import Optimizer

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    #identity matrix
    I = torch.eye(X.size(-1), device=X.device, dtype=X.dtype)
    # Perform the NS iterations
    for _ in range(steps):
        # === Complete the code

        Y = X.mT @ X
        Y_sq = Y @ Y
        update_poly = I.mul(c).add_(Y.mul(b)).add_(Y_sq.mul(a))
        X = X @ update_poly

        # === Complete the code
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    # momentum update, please consider the nesterov as True
    # ===== Complete the code =====
    grad_sign = torch.sign(grad)
    momentum.mul_(beta).add_(grad_sign, alpha=1 - beta)
    if nesterov:
        update = beta * momentum + grad_sign
    else:
        update = momentum

    # ===== Complete the code =====
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

def adam_update(grad, buf1, buf2, step, betas, eps):
    # ===== Complete the code =====
    beta1, beta2 = betas
    
    # Update moments (your version is better)
    buf1.mul_(beta1).add_(grad, alpha=1 - beta1)
    buf2.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    
    # Bias correction
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    # Compute update (your version is more numerically stable)
    denom = (buf2.sqrt() / math.sqrt(bias_correction2)).add_(eps)
    update = (buf1 / bias_correction1) / denom

    # ===== Complete the code =====
    return update

class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"],
                                         group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
    
# Try different configs and make the training loss less than 1 at the end of learning. 
# All configs can be changed
# ===== Complete the code =====
class Config:
    num_training_steps = 6000
    # total batch size = per_device_train_batch_size * gradient_accumulation_steps, 
    # I suggest you to set the total batch size no less than 32, 
    # for example, per_device_train_batch_size = 2, gradient_accumulation_steps = 16
    per_device_train_batch_size = 12
    gradient_accumulation_steps = 6
    learning_rate = 3.5e-3
    weight_decay = 1e-3
    warmup_steps = 200
    logging_steps= 10
    remove_unused_columns=True
    dataloader_num_workers=2 # I suggest 4, you can try other values
    seed= 42
    # I suggest bf16, you can try fp16, but do not use both. bf16 actually not work for some GPUs, in that case, you may need to use fp16.
    # I conduct my own experiments on bf16, and it works for me. However, I have not tested fp16. So if there are any problem with fp16, please let me know.
    fp16=False
    bf16=True
    # I suggest 512 if you have enough GPU memory, 256 if you have limited GPU memory
    max_length= 128
    max_grad_norm=1.0 # I suggest 1.0, you can try other values
    report_to="none", # optional, if you want to use wandb it is also ok
    run_name="assignment5",
    gradient_checkpointing=True, # optional, enable to save GPU memory, but if you have enough GPU memory, you can disable it.
    dataloader_drop_last=False, # Optional: if the last batch is not full, set it to True
    include_num_input_tokens_seen=True,
# ===== Complete the code =====

def set_random_seed(random_seed: int = 42):
    # Setting random seed for all
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    logger.info(f"Set random seed to {random_seed}")
    return random_seed

config = Config()
assert not (config.bf16 and config.fp16), "fp16 and bf16 cannot be used together"
# Set seeds
set_random_seed(config.seed)
# Load model and tokenizer
raw_train_dataset = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)


# load the model and tokenizer, we train the model from scratch!
# you are not allowed to load the model from pretrained.
model_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B") # you can also use other open-source model.
model = AutoModelForCausalLM.from_config(model_config) # you can also use other open-source model.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") # you can also use other open-source model.
# Tokenization function
def tokenize_function(examples):
    texts = examples["text"]
    return tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=config.max_length,
        return_attention_mask=True,
        return_special_tokens_mask=False,
    )
train_dataset = raw_train_dataset.map(
    tokenize_function, 
    batched=True,     
    remove_columns=raw_train_dataset.column_names
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
    return_tensors="pt",
)
print(model)


# Build the Muon optimizer
# ===== Complete the code =====
muon_params_list = []
adam_params_list = []
patterns = [
    "emb",
    "norm",
    "lm_head",
    "bias",
    "wte",
    "wpe",
    "output",
    "conv",
    "rotary",
]
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    # pick out those parameters that are not in the patterns and have 2-D shape, we only apply the muon optimizer to those parameters, while leave the rest of the parameters to adam optimizer
    # ===== Complete the code =====
    use_muon = (param.ndim == 2 and not any(p in name for p in patterns))
    # ===== Complete the code =====
    param.use_muon = use_muon
    if use_muon:
        muon_params_list.append(param)
    else:
        adam_params_list.append(param)

params_groups = [
    {
        "params": muon_params_list,
        "use_muon": True,
        "lr": config.learning_rate,
        "momentum": 0.95,
        "weight_decay": config.weight_decay,
    },
    
    {
        "params": adam_params_list,
        "use_muon": False,
        "lr": config.learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": config.weight_decay,
    },
]

from transformers import get_scheduler
optimizer = SingleDeviceMuonWithAuxAdam(params_groups)
# you can also use other scheduler
scheduler = get_scheduler(
    "cosine", 
    optimizer,
    num_warmup_steps=config.warmup_steps,
    num_training_steps=config.num_training_steps,
)

# Setup training arguments
# from transformers import ProgressCallback
# from transformers.integrations import NotebookProgressCallback
training_args = TrainingArguments(
    max_steps=config.num_training_steps,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    logging_steps=config.logging_steps,
    remove_unused_columns=config.remove_unused_columns,
    dataloader_num_workers=config.dataloader_num_workers,
    max_grad_norm=config.max_grad_norm,
    seed=config.seed,
    fp16=config.fp16,
    bf16=config.bf16,
    tf32=True,
    report_to=config.report_to[0],
    run_name="Assignment5",
    gradient_checkpointing=config.gradient_checkpointing,
    dataloader_drop_last=config.dataloader_drop_last,                    # Optional: if the last batch is not full, set it to True
    include_num_input_tokens_seen=config.include_num_input_tokens_seen,
    dataloader_persistent_workers=True,
    logging_strategy="steps",
)

# Data collator
# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler)
)
# trainer.remove_callback(ProgressCallback)
# trainer.add_callback(NotebookProgressCallback)
# Start training
logger.info("Starting training...")
logger.info(f"Training steps: {config.num_training_steps}")
logger.info(f"Batch size: {config.per_device_train_batch_size}")
logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
logger.info(f"Learning rate: {config.learning_rate}")

# if config.resume_from_checkpoint:
#     logger.info(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
#     trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
# else:
trainer.train()
logger.info("Training completed!")