import os 
import json
from PIL import Image 
import numpy as np 
from tqdm import tqdm
import math 
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from guidance.sd import StableDiffusion
from utils import *


def seed_everything(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)


def init_model(args):
    model = StableDiffusion(args, t_range=[0.02, 0.98])
    return model
    
    
def run(args):
    args.precision = torch.float16 if args.precision == "fp16" else torch.float32
    
    # Initialize model and optimizing parameters
    model = init_model(args)
    device = model.device
    guidance_scale = args.guidance_scale
    steps = args.step
    
    # Get text embeddings
    cond_embeddings = model.get_text_embeds(args.prompt)
    uncond_embeddings = model.get_text_embeds(args.negative_prompt)
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    
    # Initialize latents with random Gaussian for generation
    latents = nn.Parameter(
        torch.randn(1, 4, 64, 64, device=device, dtype=args.precision)
    )
    
    # Setup optimizer
    if args.loss_type == "vsd":
        # VSD needs to optimize both latents and LoRA parameters
        optimizer = torch.optim.AdamW(
            [
                {'params': [latents], 'lr': args.lr},
                {'params': model.lora_layers, 'lr': args.lora_lr}  # lora_layers is now a list
            ],
            weight_decay=0
        )
    else:
        # SDS and SDI only optimize latents
        optimizer = torch.optim.AdamW([latents], lr=args.lr, weight_decay=0)

    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(steps*1.5))

    # Run optimization
    for step in tqdm(range(steps)):
        optimizer.zero_grad()
        
        if args.loss_type == "sds":
            loss = model.get_sds_loss(
                latents=latents,
                text_embeddings=text_embeddings, 
                guidance_scale=guidance_scale,
            )
            
        elif args.loss_type == "sdi":
            loss = model.get_sdi_loss(
                latents=latents,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale,
                current_iter=step,
                total_iters=steps
            )
            
        elif args.loss_type == "vsd":
            loss = model.get_vsd_loss(
                latents=latents,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale,
                lora_loss_weight=args.lora_loss_weight,
            )
            
        else:
            raise ValueError("Invalid loss type")
        
        # Backward
        loss.backward()

        # Update optimizers
        optimizer.step()
        scheduler.step()
        
        # Logging
        if step % args.log_step == 0:
            with torch.no_grad():
                img = model.decode_latents(latents)
                img_save_path = os.path.join(args.save_dir, f"{step}.png")
                torch_to_pil(img).save(img_save_path)
                print(f"Step: {step}, Loss: {loss.item():.6f}")
            
    # Final save
    img = model.decode_latents(latents)
    prompt_key = args.prompt.replace(" ", "_")
    img_save_path = os.path.join(args.save_dir, f"{prompt_key}.png")
    torch_to_pil(img).save(img_save_path)
    print(f"Save path: {img_save_path}")
        
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="low quality")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    
    parser.add_argument("--loss_type", type=str, default="sds", choices=["sds", "sdi", "vsd"])
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--step", type=int, default=500)
    
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate for latents")
    
    # VSD specific parameters
    parser.add_argument("--lora_lr", type=float, default=1e-4, 
                        help="Learning rate for LoRA in VSD")
    parser.add_argument("--lora_loss_weight", type=float, default=1.0, 
                        help="Weight for LoRA denoising loss in VSD")
    parser.add_argument("--lora_rank", type=int, default=4, 
                    help="LoRA rank for VSD (default: 4)")
    
    parser.add_argument("--log_step", type=int, default=25)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"])
    
    return parser.parse_args()

    
def main():
    args = parse_args()
    
    if os.path.exists(args.save_dir):
        print("[*] Save directory already exists. Overwriting...")
    else:
        os.makedirs(args.save_dir)
        
    log_opt = vars(args)
    config_path = os.path.join(args.save_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(log_opt, f, indent=4)
    
    print(f"[*] Running {args.loss_type}")
    print(f"[*] Prompt: {args.prompt}")
    print(f"[*] Guidance scale: {args.guidance_scale}")
    print(f"[*] Steps: {args.step}")

    run(args)
    

if __name__ == "__main__":
    main()