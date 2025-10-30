from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from peft import LoraConfig


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] Loading Stable Diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae.eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet.eval()
        
        # Freeze models
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        
        # Schedulers
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )
        self.inverse_scheduler.alphas_cumprod = self.inverse_scheduler.alphas_cumprod.to(self.device)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod
        
        print(f'[INFO] Loaded Stable Diffusion!')
        
        # Initialize VSD components if needed
        if args.loss_type == "vsd":
            self._init_vsd_components(args.lora_rank)
    
    def _init_vsd_components(self, lora_rank=4):
        """Initialize LoRA for VSD"""
        print(f"[INFO] Initializing VSD with LoRA rank={lora_rank}")
        
        self.unet.requires_grad_(False)
        
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        self.unet.add_adapter(unet_lora_config)
        self.lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """Get text embeddings from prompt"""
        inputs = self.tokenizer(
            prompt, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt'
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=7.5):
        """
        Predict noise with Classifier-Free Guidance (CFG)
        
        CFG formula: noise_pred = uncond + guidance_scale * (cond - uncond)
        """
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)
        
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        
        # Classifier-Free Guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred
    
    def get_sds_loss(self, latents, text_embeddings, guidance_scale=7.5):
        """
        Score Distillation Sampling (SDS) Loss
        
        Reference: DreamFusion (https://arxiv.org/abs/2209.14988)
        """
        # TODO: Implement SDS loss
        # Get random timestep
        t = torch.randint(1, self.num_train_timesteps, (1,), device=self.device)
        eps = torch.randn_like(latents).to(self.device)
        alpha_bar_t = self.alphas[t]
        # xt = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        xt = self.scheduler.add_noise(
            original_samples=latents,   # x0 in latent space
            noise=eps,
            timesteps=t
        )
        # Get Noise Residual
        noise_pred = self.get_noise_preds(xt, t, text_embeddings, guidance_scale=guidance_scale)
        noise_residual = noise_pred - eps

        # w(t)
        w = (1.0 - alpha_bar_t)

        g = w * noise_residual  
        g = torch.nan_to_num(g)
        target = (latents - g).detach()
        return 0.5 * nn.functional.mse_loss(latents, target)
    
    def get_vsd_loss(self, latents, text_embeddings, guidance_scale=7.5, lora_loss_weight=1.0):
        """
        Variational Score Distillation (VSD) Loss
        
        Reference: ProlificDreamer (https://arxiv.org/abs/2305.16213)
        """
        # TODO: Implement VSD loss
        B = latents.shape[0]
        t = torch.randint(self.min_step, self.max_step + 1, (B,), dtype=torch.long, device=self.device)
        eps = torch.randn_like(latents).to(self.device)
        xt = self.scheduler.add_noise(original_samples=latents, noise=eps, timesteps=t)
        w = (1.0 - self.alphas[t])
        alpha_t = self.alphas[t] 

        with torch.no_grad():
            self.unet.disable_adapters()
            noise_pred = self.get_noise_preds(xt, t, text_embeddings, guidance_scale=guidance_scale)
            self.unet.enable_adapters()
            # noise_pred_lora = self.get_noise_preds(xt, t, text_embeddings, guidance_scale=guidance_scale)
            latent_model_input = torch.cat([xt] * 2)
            tt = torch.cat([t] * 2)
            lora_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
            lora_pred = latent_model_input * torch.cat([w] * 2, dim=0).view(-1, 1, 1, 1) + noise_pred * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
            lora_pred_uncond, lora_pred_pos = lora_pred.chunk(2)
            noise_pred_lora = lora_pred_uncond + guidance_scale * (lora_pred_pos - lora_pred_uncond)

        noise_residual = noise_pred - noise_pred_lora
    
        g = torch.nan_to_num(w * noise_residual)
        target = (latents - g).detach()
        vsd_loss = 0.5 * nn.functional.mse_loss(latents, target, reduction="mean")

        # lora loss
        """B = latents.shape[0]
        t = torch.randint(self.min_step, self.max_step + 1, (B,), dtype=torch.long, device=self.device)
        eps = torch.randn_like(latents).to(self.device)
        xt = self.scheduler.add_noise(original_samples=latents.detach(), noise=eps, timesteps=t)"""

        latent_model_input = torch.cat([xt] * 2)
        tt = torch.cat([t] * 2)
        lora_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
        lora_pred = latent_model_input * torch.cat([w] * 2, dim=0).view(-1, 1, 1, 1) + noise_pred * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
        dump, lora_pred = lora_pred.chunk(2)

        lora_pred = alpha_t * lora_pred
        target = alpha_t * eps
        lora_loss = 0.5 * nn.functional.mse_loss(lora_pred, target, reduction="mean")
        
        loss = vsd_loss + lora_loss_weight * lora_loss
        return loss
    
    @torch.no_grad()
    def invert_noise(self, latents, target_t, text_embeddings, guidance_scale=-7.5, n_steps=10, eta=0.3):
        """
        DDIM Inversion: x0 -> x_t
        
        Inverts clean latents (x0) to noisy latents (x_t) using DDIM inversion.
        
        Args:
            latents: Clean latents x0
            target_t: Target timestep to invert to
            text_embeddings: Text condition
            guidance_scale: CFG scale (typically negative for inversion!)
            n_steps: Number of inversion steps
            eta: Noise level for stochasticity
            
        Returns:
            Inverted noisy latents x_t
        """
        # TODO: (Implement DDIM inversion by yourself — do NOT call built-in inversion helpers):
        # --------------------------------------------------------------------
        # Write your own DDIM inversion loop that maps x0 -> x_t at `target_t`.
        # You may *read* external implementations for reference, but you must
        # NOT call any "invert"/"ddim_invert"/"invert_step" utilities
        # from diffusers or other libraries.
        B = latents.shape[0]
        timesteps = torch.linspace(0, target_t.item(), n_steps + 1, device=self.device).long()

        noisy_latents = latents
        for i in range(len(timesteps)-1):
            t = torch.full((B,), timesteps[i], device=self.device, dtype=torch.long)
            noise_pred = self.get_noise_preds(noisy_latents, t, text_embeddings, guidance_scale)
            alpha_bar_t = self.alphas[timesteps[i]]
            alpha_bar_t_next = self.alphas[timesteps[i+1]]
            #x0_pred = (noisy_latents - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqert()
            print((1 - alpha_bar_t_next))
            print(1 - alpha_bar_t)
            print(((1 - alpha_bar_t_next) / (1 - alpha_bar_t)).sqrt())
            sigma_t = eta                                                   \
                    * ((1 - alpha_bar_t_next) / (1 - alpha_bar_t)).sqrt()   \
                    * (1 - alpha_bar_t / alpha_bar_t_next).sqrt()
            dir_xt = ((1-alpha_bar_t_next) - sigma_t**2).sqrt() * noise_pred
            # compute x_t without "random noise"
            noisy_latents = alpha_bar_t_next.sqrt() * noise_pred + dir_xt
            # Add noise to the sample
            noise = torch.randn_like(noisy_latents)
            noisy_latents += sigma_t * noise
        
        return noisy_latents
    
    def get_sdi_loss(
        self, 
        latents,                    
        text_embeddings,            
        guidance_scale=7.5,         
        current_iter=0,             
        total_iters=500,            
        inversion_guidance_scale=-7.5,  
        inversion_n_steps=10,       
        inversion_eta=0.3,          
        update_interval=25,        
    ):
        """
        Score Distillation via Inversion (SDI) Loss
        
        Reference: Score Distillation via Reparametrized DDIM (https://arxiv.org/abs/2405.15891)
        
        Key Insight: Instead of using random noise like SDS, SDI uses DDIM inversion
        to get better noise that's consistent with the current latents.
        
        Strategy:
        1. Timestep annealing: t decreases from max_step to min_step during training
        2. Periodically update target via DDIM inversion
        3. Use MSE loss between current latents and cached target
        
        Args:
            latents: Current optimized latents (B, 4, H, W)
            text_embeddings: Concatenated [uncond, cond] embeddings (2*B, seq_len, dim)
            guidance_scale: CFG scale for final denoising step
            current_iter: Current training iteration number
            total_iters: Total number of training iterations
            inversion_guidance_scale: CFG scale for DDIM inversion (typically negative)
            inversion_n_steps: Number of inversion steps from x0 to x_t
            inversion_eta: Stochasticity level for DDIM inversion (0=deterministic)
            update_interval: Update cached target every N iterations
            
        Returns:
            loss: MSE loss between current latents and cached target
        """
        B = latents.shape[0]
        
        # TODO: Create current timestep tensor based on training progress
        # t = ...
        B = latents.shape[0]
        progress = current_iter / (total_iters - 1)
        t_value = self.max_step - progress * (self.max_step - self.min_step)
        t = torch.full((B,), int(t_value), device=self.device, dtype=torch.long)
        w = (1.0 - self.alphas[t])
        
        # Check if we need to update target
        should_update = (current_iter % update_interval == 0) or not hasattr(self, 'sdi_target')
        
        if should_update:
            with torch.no_grad():
                # Perform DDIM inversion: x0 -> x_t
                latents_noisy = self.invert_noise(
                    latents, t, text_embeddings,
                    guidance_scale=inversion_guidance_scale,
                    n_steps=inversion_n_steps,
                    eta=inversion_eta
                )
                
                # TODO: Predict noise from inverted noisy latents
                # noise_pred = ...
                noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings, guidance_scale)

                # TODO: Denoise to get target x0 using predicted noise
                # target = ... = (x_t - √(1-α_t) * ε_θ) / √α_t
                alpha_bar_t = self.alphas[t]
                target = (latents_noisy - (1 - alpha_bar_t).sqrt()*noise_pred) / alpha_bar_t.sqrt()
                # Cache the target
                self.sdi_target = target.detach()
        
        # TODO: Compute MSE loss between current latents and cached target
        # loss = ...
        loss = 0.5 * nn.functional.mse_loss(latents, self.sdi_target, reduction="mean")
        return loss
        
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to RGB images"""
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        """Encode RGB images to latents"""
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents