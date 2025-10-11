from diffusers import DDIMScheduler, StableDiffusionPipeline
from peft import LoraConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        print(f'[INFO] loaded stable diffusion!')

        if args.loss_type == "vsd":
            # Initialize VSD components
            self._init_vsd_components(args.lora_rank)

    def _init_vsd_components(self, lora_rank=4):
        """
        Initialize LoRA components for VSD using peft LoraConfig
        
        Args:
            lora_rank: LoRA rank (default 4)
        """
        print(f"[INFO] Initializing VSD with LoRA rank={lora_rank}")
        
        # Freeze the original UNet
        self.unet.requires_grad_(False)
        
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,  
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # attention layers
        )
        
        # Add LoRA adapters to UNet
        self.unet.add_adapter(unet_lora_config)
        self.lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
    

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        """Get noise predictions from UNet (with current LoRA settings)"""
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)
        
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred

    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        # TODO: Implement the loss function for SDS
        raise NotImplementedError("SDS is not implemented yet.")
    
    def get_sdi_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
        current_iter=0,
        total_iters=500,
    ):
        # TODO: Implement the loss function for SDI
        raise NotImplementedError("SDI is not implemented yet.")
        
    def get_vsd_loss(
        self,
        latents,
        text_embeddings,
        guidance_scale=7.5,
        lora_loss_weight=1.0,
    ):
        # TODO: Implement the loss function for VSD
        raise NotImplementedError("VSD is not implemented yet.")
        
    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents