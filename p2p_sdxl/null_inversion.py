from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers import DDIMScheduler

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DDIMScheduler,
)
from tqdm import tqdm
from torch.optim.adam import Adam



GUIDANCE_SCALE = 7.5


class SDXLDDIMPipeline(StableDiffusionXLImg2ImgPipeline):
    def get_noise_pred_single(self, latents, t, 
                                  encoder_hidden_states, 
                                  added_cond_kwargs):
        noise_pred = self.unet(latents, t, 
                                  encoder_hidden_states=encoder_hidden_states,
                                  added_cond_kwargs=added_cond_kwargs)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None, added_cond_kwargs=None):
        latents_input = torch.cat([latents] * 2)
        # if context is None:
        #     context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context,
                                                 added_cond_kwargs=added_cond_kwargs)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def null_optimization(self, latents, num_inference_steps, epsilon, negative_prompt_embeds, prompt_embeds, added_cond_kwargs):
        uncond_embeddings = torch.zeros_like(negative_prompt_embeds)
        cond_embeddings = prompt_embeds
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        num_inner_steps = 2
        bar = tqdm(total=num_inner_steps * num_inference_steps)
        for i in range(num_inference_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.scheduler.timesteps[i]

            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings, added_cond_kwargs)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings, added_cond_kwargs)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                
                loss = F.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
          

                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
                latent_cur = self.get_noise_pred(latent_cur, t, False, context, added_cond_kwargs)
        bar.close()

        return uncond_embeddings_list
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        with torch.no_grad():
            self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)

            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                prompt_2,
                strength,
                num_inference_steps,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
            )

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = True
            # 3. Encode input prompt
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None)
                if cross_attention_kwargs is not None
                else None
            )
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )

            # 4. Preprocess image
            image = self.image_processor.preprocess(image)

            self.scheduler.set_timesteps(num_inference_steps, device=device)
        
            # 6. Prepare latent variables
            latents = self.prepare_latents(
                image,
                None,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                False,
            )
        
            height, width = latents.shape[-2:]
            height = height * self.vae_scale_factor
            width = width * self.vae_scale_factor

            original_size = original_size or (height, width)
            target_size = target_size or (height, width)

            # 8. Prepare added time ids & embeddings
            if negative_original_size is None:
                negative_original_size = original_size
            if negative_target_size is None:
                negative_target_size = target_size

            add_text_embeds = pooled_prompt_embeds
            if self.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

            add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                aesthetic_score,
                negative_aesthetic_score,
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            prev_timestep = None

            with torch.no_grad():
                all_latent = [latents]
                latent = latents.clone().detach()

                for i in range(num_inference_steps):
                    latent_model_input = latents
                    t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
                    noise_pred = self.get_noise_pred_single(latent_model_input, t, 
                                    encoder_hidden_states=prompt_embeds,  
                                    added_cond_kwargs=added_cond_kwargs)

                    latents = self.next_step(noise_pred, t, latents)
                    all_latent.append(latents)
            
            ddim_latents = all_latent

        early_stop_epsilon = 1e-6
        uncond_embeddings = self.null_optimization(ddim_latents, num_inference_steps, early_stop_epsilon, negative_prompt_embeds, prompt_embeds, added_cond_kwargs)
        torch.cuda.empty_cache() 
        return ddim_latents[-1].to(torch.float16), uncond_embeddings
        # return StableDiffusionXLPipelineOutput(images=ddim_latents)
   
