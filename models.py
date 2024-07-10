import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat
import time
from tools.torch_tools import wav_to_fbank, sinusoidal_positional_embedding

from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from audioldm.utils import default_audioldm_config, get_metadata

from transformers import CLIPTokenizer, AutoTokenizer, T5Tokenizer
from transformers import CLIPTextModel, T5EncoderModel, AutoModel
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from transformers import CLIPProcessor, CLIPModel

import diffusers
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers import AutoencoderKL as DiffuserAutoencoderKL
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, RandomResizedCrop
from diffusers import AudioLDMPipeline

def build_pretrained_models(name):
    checkpoint = torch.load(name, map_location="cpu")
    scale_factor = checkpoint["state_dict"]["scale_factor"].item()

    vae_state_dict = {k[18:]: v for k, v in checkpoint["state_dict"].items() if "first_stage_model." in k}

    config = default_audioldm_config(name)
    vae_config = config["model"]["params"]["first_stage_config"]["params"]
    vae_config["scale_factor"] = scale_factor

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(vae_state_dict)

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    vae.eval()
    fn_STFT.eval()
    return vae, fn_STFT


class EffNetb3(nn.Module):
    def __init__(self, pretrained_model_path, embedding_dim=1024, pretrained=True):
        super(EffNetb3, self).__init__()
        self.model_name = 'effnetb3'
        self.pretrained = pretrained
        # Create model
        # self.effnet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b3', pretrained=self.pretrained)
        # torch.save(self.effnet, 'model.pth')
        self.effnet = torch.hub.load(pretrained_model_path, 'efficientnet_b3', trust_repo=True, source='local')
        #self.effnet.conv_stem = nn.Conv2d(1, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.embedder = nn.Conv2d(384, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #out = self.effnet(x)
        out = self.effnet.conv_stem(x)
        out = self.effnet.bn1(out)
        out = self.effnet.act1(out)
        for i in range(len(self.effnet.blocks)):
            out = self.effnet.blocks[i](out)
        out = self.embedder(out)
        return out


class EffNetb3_last_layer(nn.Module):
    def __init__(self, pretrained_model_path, embedding_dim=1024, pretrained=True):
        super(EffNetb3_last_layer, self).__init__()
        self.model_name = 'effnetb3'
        self.pretrained = pretrained
        self.effnet = torch.hub.load(pretrained_model_path, 'efficientnet_b3', trust_repo=True, source='local')
        self.effnet.classifier = nn.Linear(1536, embedding_dim)

    def forward(self, x):
        out = self.effnet(x)
        return out.unsqueeze(-1)


class Clip4Video(nn.Module):
    def __init__(self, model, embedding_dim=1024, pretrained=True, pe=False):
        super(Clip4Video, self).__init__()
        self.pretrained = pretrained
        self.clip_vision = CLIPVisionModelWithProjection.from_pretrained(model)
        self.clip_text = CLIPTextModelWithProjection.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        input_dim = 512 if "clip-vit-base" in model else 768
        self.linear_layer = nn.Linear(input_dim, embedding_dim)
        self.pe = sinusoidal_positional_embedding(30, input_dim) if pe else None
        print("*****PE*****") if pe else print("*****W/O PE*****")

    def forward(self, text=None, image=None, video=None):
        assert text is not None or image is not None or video is not None, "At least one of text, image or video should be provided"
        if text is not None and video is None:
            inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=77).to(self.clip_text.device)
            out = self.clip_text(**inputs)
            out = out.text_embeds.repeat(20, 1)
        elif video is not None and text is None:
            out = self.clip_vision(video.to(self.clip_vision.device))          # input video x: t * 3 * w * h
            out = out.image_embeds        # t * 512
            if self.pe is not None:
                out = out + self.pe[:out.shape[0], :].to(self.clip_vision.device)
            # out['last_hidden_state'].shape # t * 50 * 768
            # out['image_embeds'].shape      # t * 512
        elif text is not None and video is not None:
            text_inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=77).to(self.clip_text.device)
            video_out = self.clip_vision(video.to(self.clip_vision.device))
            video_out = video_out.image_embeds
            text_out = self.clip_text(**text_inputs)
            text_out = text_out.text_embeds.repeat(video_out.shape[0], 1)
            # out = text_out + video_out
            # concat 
            out = torch.cat([text_out, video_out], dim=0)
        out = self.linear_layer(out)     # t * 1024
        return out


class AudioDiffusion(nn.Module):
    def __init__(
        self,
        fea_encoder_name,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,
        img_pretrained_model_path=None,
        task=None,
        embedding_dim=1024,
        pe=False
    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.fea_encoder_name = fea_encoder_name
        self.scheduler_name = scheduler_name
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.freeze_text_encoder = freeze_text_encoder
        self.uncondition = uncondition
        self.task = task
        self.pe = pe

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        self.inference_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")

        if unet_model_config_path:
            unet_config = UNet2DConditionModel.load_config(unet_model_config_path)
            print("unet_config", unet_config)
            self.unet = UNet2DConditionModel.from_config(unet_config, subfolder="unet")
            self.set_from = "random"
            print("UNet initialized randomly.")
        else:
            self.unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder="unet")
            self.set_from = "pre-trained"
            self.group_in = nn.Sequential(nn.Linear(8, 512), nn.Linear(512, 4))
            self.group_out = nn.Sequential(nn.Linear(4, 512), nn.Linear(512, 8))
            print("UNet initialized from stable diffusion checkpoint.")

        if self.task == "text2audio":
            if "stable-diffusion" in self.fea_encoder_name:
                self.tokenizer = CLIPTokenizer.from_pretrained(self.fea_encoder_name, subfolder="tokenizer")
                self.text_encoder = CLIPTextModel.from_pretrained(self.fea_encoder_name, subfolder="text_encoder")
            elif "t5" in self.fea_encoder_name and "Chinese" not in self.fea_encoder_name:
                self.tokenizer = AutoTokenizer.from_pretrained(self.fea_encoder_name)
                self.text_encoder = T5EncoderModel.from_pretrained(self.fea_encoder_name)
            elif "Chinese" in self.fea_encoder_name:
                self.tokenizer = T5Tokenizer.from_pretrained(self.fea_encoder_name)
                self.text_encoder = T5EncoderModel.from_pretrained(self.fea_encoder_name)
            elif "clap" in self.fea_encoder_name:
                self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
                self.CLAP_model = laion_clap.CLAP_Module(enable_fusion=False)
                self.CLAP_model.load_ckpt(self.fea_encoder_name)
            elif "clip-vit" in self.fea_encoder_name:
                # self.CLIP_model = CLIPModel.from_pretrained(self.fea_encoder_name)
                # self.CLIP_processor = CLIPProcessor.from_pretrained(self.fea_encoder_name)
                self.CLIP_model = CLIPTextModelWithProjection.from_pretrained(self.fea_encoder_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.fea_encoder_name)
                if "base" in self.fea_encoder_name:
                    self.linear_layer = nn.Linear(512, embedding_dim)
                else:
                    self.linear_layer = nn.Linear(768, embedding_dim)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.fea_encoder_name)
                self.text_encoder = AutoModel.from_pretrained(self.fea_encoder_name)
        elif self.task == "image2audio":
            if "clip-vit" in self.fea_encoder_name:
                self.CLIP_model = CLIPModel.from_pretrained(self.fea_encoder_name)
                self.CLIP_processor = CLIPProcessor.from_pretrained(self.fea_encoder_name)
                self.linear_layer = nn.Linear(512, embedding_dim)
            # self.img_fea_extractor = EffNetb3(img_pretrained_model_path)
            else:
                self.img_fea_extractor = EffNetb3_last_layer(img_pretrained_model_path)
        elif self.task == "video2audio":
            self.vid_fea_extractor = Clip4Video(model=self.fea_encoder_name, embedding_dim=embedding_dim, pe=pe)

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        if self.freeze_text_encoder:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
        else:
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def encode_text_CLAP(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        if self.freeze_text_encoder:
            with torch.no_grad():
                encoder_hidden_states = self.CLAP_model.model.get_text_embedding(prompt)
        else:
            encoder_hidden_states = self.CLAP_model.model.get_text_embedding(prompt)

        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def encode_image(self, prompt, device):
        if "clip-vit" in self.fea_encoder_name:
            with torch.no_grad():
                inputs = self.CLIP_processor(text=["aaa"], images=prompt, return_tensors="pt", padding=True).to(device)
                encoder_hidden_states = self.CLIP_model(**inputs).image_embeds
            encoder_hidden_states = self.linear_layer(encoder_hidden_states)    # b * 1024
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1).to(device)
        else:
            img_fea = self.img_fea_extractor(prompt) 
            encoder_hidden_states = img_fea.view(img_fea.shape[0], img_fea.shape[1], -1).permute(0, 2, 1)
        boolean_encoder_mask = torch.ones((encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]), dtype=torch.bool)
        boolean_encoder_mask = boolean_encoder_mask.to(device)

        return encoder_hidden_states, boolean_encoder_mask
    
    def encode_video(self, video_batch, text=None, device=None):
        vid_feas = []
        for i, video in enumerate(video_batch):
            if text:
                vid_fea = self.vid_fea_extractor(video=video, text=text[i]) # t * fea_dim
            else:
                vid_fea = self.vid_fea_extractor(video=video)
            vid_feas.append(vid_fea)
        
        padding = 0
        size = max(v.size(0) for v in vid_feas)
        batch_size = len(vid_feas)
        embed_size = vid_feas[0].size(1)
        encoder_hidden_states = vid_feas[0].new(batch_size, size, embed_size).fill_(padding)
        boolean_encoder_mask = torch.ones((batch_size, size), dtype=torch.bool)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(vid_feas):
            copy_tensor(v, encoder_hidden_states[i][: len(v)])
            boolean_encoder_mask[i, len(v):] = False    
        return encoder_hidden_states.to(device), boolean_encoder_mask.to(device)

    def encode_text_CLIP(self, prompt, device):
        # tmp_image = np.ones((512, 512, 3))
        # with torch.no_grad():
        #     inputs = self.CLIP_processor(text=prompt, images=tmp_image, return_tensors="pt", padding=True, max_length=77, truncation=True).to(device)
        #     encoder_hidden_states = self.CLIP_model(**inputs).text_embeds   # b * 768
        text_inputs = self.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", max_length=77).to(device)
        encoder_hidden_states = self.CLIP_model(**text_inputs).text_embeds
        encoder_hidden_states = self.linear_layer(encoder_hidden_states)    # b * 1024
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).to(device)
        boolean_encoder_mask = torch.ones((encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]), dtype=torch.bool)
        boolean_encoder_mask = boolean_encoder_mask.to(device)

        return encoder_hidden_states, boolean_encoder_mask

    def forward(self, latents, text=None, video=None, image=None, validation_mode=False, device=None):
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)
        # encoder_hidden_states.shape  [b, t, f]
        if self.task == "text2audio":
            if "clip-vit" in self.fea_encoder_name:
                encoder_hidden_states, boolean_encoder_mask = self.encode_text_CLIP(text, device)
            else:
                encoder_hidden_states, boolean_encoder_mask = self.encode_text(text)
            if self.uncondition:
                mask_indices = [k for k in range(len(text)) if random.random() < 0.1]
                # mask_indices = [k for k in range(len(prompt))]
                if len(mask_indices) > 0:
                    encoder_hidden_states[mask_indices] = 0
        elif self.task == "image2audio":
            encoder_hidden_states, boolean_encoder_mask = self.encode_image(image, device=device)
        elif self.task == "video2audio":
            encoder_hidden_states, boolean_encoder_mask = self.encode_video(video, text, device=device)
        
        bsz = latents.shape[0]
        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            # Sample a random timestep for each instance
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.set_from == "random":
            model_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states, 
                encoder_attention_mask=boolean_encoder_mask
            ).sample

        elif self.set_from == "pre-trained":
            compressed_latents = self.group_in(noisy_latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            model_pred = self.unet(
                compressed_latents, timesteps, encoder_hidden_states, 
                encoder_attention_mask=boolean_encoder_mask
            ).sample
            model_pred = self.group_out(model_pred.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def inference(self, inference_scheduler, text=None, video=None, image=None, num_steps=20, guidance_scale=3, num_samples_per_prompt=1, 
                  disable_progress=True, device=None):
        start = time.time()
        classifier_free_guidance = guidance_scale > 1.0

        #print("ldm time 0", time.time()-start, prompt)
        if self.task == "text2audio":
            batch_size = len(text) * num_samples_per_prompt

            if classifier_free_guidance:
                if "clip-vit" in self.fea_encoder_name:
                    encoder_hidden_states, boolean_encoder_mask = self.encode_text_clip_classifier_free(text, num_samples_per_prompt, device=device)
                else:
                    encoder_hidden_states, boolean_encoder_mask = self.encode_text_classifier_free(text, num_samples_per_prompt)
            else:
                encoder_hidden_states, boolean_encoder_mask = self.encode_text(text)
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_samples_per_prompt, 0)
                boolean_encoder_mask = boolean_encoder_mask.repeat_interleave(num_samples_per_prompt, 0)
        elif self.task == "image2audio":
            if classifier_free_guidance:
                encoder_hidden_states, boolean_encoder_mask = self.encode_image_classifier_free(image, num_samples_per_prompt, device=device)
            else:
                encoder_hidden_states, boolean_encoder_mask = self.encode_image_no_grad(image, device=device)
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_samples_per_prompt, 0)
                boolean_encoder_mask = boolean_encoder_mask.repeat_interleave(num_samples_per_prompt, 0)
        elif self.task == "video2audio":
            batch_size = len(video) * num_samples_per_prompt
            encoder_hidden_states, boolean_encoder_mask = self.encode_video_classifier_free(video, text, num_samples_per_prompt, device=device)
        # import pdb;pdb.set_trace()
        #print("ldm time 1", time.time()-start)
        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, encoder_hidden_states.dtype, device)
        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        #print("ldm time 2", time.time()-start, timesteps)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

            #print("ldm emu", i, time.time()-start)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=boolean_encoder_mask
            ).sample

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)

        #print("ldm time 3", time.time()-start)
        if self.set_from == "pre-trained":
            latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return latents

    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 256, 16)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]
                
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""] * len(prompt)

        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids.to(device)
        uncond_attention_mask = uncond_batch.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
            )[0]
                
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        # import pdb;pdb.set_trace()
        return prompt_embeds, boolean_prompt_mask

    def encode_image_no_grad(self, prompt, device):
        with torch.no_grad():
            img_fea = self.img_fea_extractor(prompt) 
        encoder_hidden_states = img_fea.view(img_fea.shape[0], img_fea.shape[1], -1).permute(0, 2, 1)
        boolean_encoder_mask = torch.ones((encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]), dtype=torch.bool)
        boolean_encoder_mask = boolean_encoder_mask.to(device)

        return encoder_hidden_states, boolean_encoder_mask
    
    def encode_text_clip_classifier_free(self, prompt, num_samples_per_prompt, device):
        # 如果想测试输入文本的效果，就用下面两行
        with torch.no_grad():
            encoder_hidden_states, boolean_encoder_mask = self.encode_text_CLIP(prompt, device)
        # if "clip-vit" in self.fea_encoder_name:
        #     with torch.no_grad():
        #         inputs = self.CLIP_processor(text=['aaa'], images=prompt, return_tensors="pt", padding=True).to(device)
        #         encoder_hidden_states = self.CLIP_model(**inputs).image_embeds   # b * 768
        #         encoder_hidden_states = self.linear_layer(encoder_hidden_states)    # b * 1024
        #         encoder_hidden_states = encoder_hidden_states.unsqueeze(1).to(device)
        #         boolean_encoder_mask = torch.ones((encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]), dtype=torch.bool)
        #         boolean_encoder_mask = boolean_encoder_mask.to(device)

        b, t, n = encoder_hidden_states.shape
        attention_mask = boolean_encoder_mask.to(device)
        prompt_embeds = encoder_hidden_states.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        negative_prompt_embeds = encoder_hidden_states.new(b, t, n).fill_(0)
        uncond_attention_mask = torch.ones((b, t), dtype=torch.bool).to(device)

        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        boolean_prompt_mask = torch.cat([uncond_attention_mask, attention_mask])

        return prompt_embeds.to(device), boolean_prompt_mask.to(device)


    def encode_image_classifier_free(self, prompt, num_samples_per_prompt, device):
        with torch.no_grad():
            if "clip-vit" in self.fea_encoder_name:
                inputs = self.CLIP_processor(text=["aaa"], images=prompt, return_tensors="pt", padding=True).to(device)
                img_fea = self.CLIP_model(**inputs).image_embeds
                img_fea = self.linear_layer(img_fea)
            else:
                img_fea = self.img_fea_extractor(prompt) 
        encoder_hidden_states = img_fea.view(img_fea.shape[0], img_fea.shape[1], -1).permute(0, 2, 1)
        b, t, n = encoder_hidden_states.shape
        boolean_encoder_mask = torch.ones((b, t), dtype=torch.bool)
        attention_mask = boolean_encoder_mask.to(device)
        prompt_embeds = encoder_hidden_states.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        negative_prompt_embeds = encoder_hidden_states.new(b, t, n).fill_(0)
        uncond_attention_mask = torch.ones((b, t), dtype=torch.bool).to(device)

        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        boolean_prompt_mask = torch.cat([uncond_attention_mask, attention_mask])

        return prompt_embeds.to(device), boolean_prompt_mask.to(device)
    
    def encode_video_classifier_free(self, video_batch, text_batch, num_samples_per_prompt, device):
        vid_feas = []
        for i, video in enumerate(video_batch):
            if text_batch:
                vid_fea = self.vid_fea_extractor(video=video.to(device), text=text_batch[i])
            else:
                vid_fea = self.vid_fea_extractor(video=video.to(device))
            vid_feas.append(vid_fea)
        
        padding = 0
        size = max(v.size(0) for v in vid_feas)
        batch_size = len(vid_feas)
        embed_size = vid_feas[0].size(1)
        encoder_hidden_states = vid_feas[0].new(batch_size, size, embed_size).fill_(padding)
        boolean_encoder_mask = torch.ones((batch_size, size), dtype=torch.bool)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(vid_feas):
            copy_tensor(v, encoder_hidden_states[i][: len(v)])
            boolean_encoder_mask[i, len(v):] = False    

        b, t, n = encoder_hidden_states.shape
        negative_prompt_embeds = encoder_hidden_states.new(b, t, n).fill_(0)
        uncond_attention_mask = torch.ones((b, t), dtype=torch.bool)

        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        encoder_hidden_states = torch.cat([negative_prompt_embeds, encoder_hidden_states])
        boolean_encoder_mask = torch.cat([uncond_attention_mask, boolean_encoder_mask])

        return encoder_hidden_states.to(device), boolean_encoder_mask.to(device)