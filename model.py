# SPDX-License-Identifier: GPL-2.0-or-later

from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
import gc
import os
import solvent.constants as constants
from typing import Optional
import torch

ddim_scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
)
lms_scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
)
pndm_scheduler = PNDMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
)


def torch_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def find_unique_filename(path: str) -> str:
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def text2image(user_input: constants.UserInput) -> Optional[str]:
    texture_name = user_input.texture_name
    texture_prompt = user_input.texture_prompt
    texture_seed = user_input.texture_seed
    model_steps = user_input.model_steps
    model_guidance_scale = user_input.model_guidance_scale
    texture_tileable = user_input.texture_tileable
    model_attention_slicing = user_input.model_attention_slicing
    model_autocast = user_input.model_autocast
    model_precision = user_input.model_precision
    model_device = user_input.model_device
    model_scheduler = user_input.model_scheduler
    texture_format = user_input.texture_format
    texture_path = user_input.texture_path

    # Overwrite the device value if CUDA/MPS is not available
    # This might go against the user's specified settings, but it's better than just throwing an error
    if model_device in ["GPU", "MPS"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_device == "MPS":
        device = "mps" if torch.backends.mps.is_available() else device
    if model_device == "CPU":
        device = "cpu"

    elif model_scheduler == "DDIM":
        scheduler = ddim_scheduler
    elif model_scheduler == "K-LMS":
        scheduler = lms_scheduler
    else:
        # By default, use the PNDM scheduler
        scheduler = pndm_scheduler

    torch_gc()

    generator = torch.Generator(device).manual_seed(texture_seed)

    if (
        constants.CURRENT_PLATFORM != "Darwin"
        and model_precision == "Half"
        and model_autocast
        and device != "cpu"
    ):
        pipe = StableDiffusionPipeline.from_pretrained(
            constants.MODEL_PATH,
            local_files_only=True,
            use_auth_token=False,
            revision="fp16",
            torch_dtype=torch.float16,
            scheduler=scheduler,
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            constants.MODEL_PATH,
            local_files_only=True,
            use_auth_token=False,
            scheduler=scheduler,
        ).to(device)

    pipe.set_progress_bar_config(disable=False)

    if model_attention_slicing:
        pipe.enable_attention_slicing()

    if texture_tileable:
        targets = [pipe.vae, pipe.unet]
        for target in targets:
            for module in target.modules():
                if isinstance(module, torch.nn.Conv2d):
                    # Patch to make tileable textures: https://gitlab.com/-/snippets/2395088
                    module.padding_mode = "circular"

    if constants.CURRENT_PLATFORM == "Darwin":
        try:
            # TODO: First-time "warmup" pass (https://github.com/huggingface/diffusers/issues/372)
            _ = pipe(texture_prompt, num_inference_steps=1)
            image = pipe(
                prompt=texture_prompt,
                num_inference_steps=model_steps,
                guidance_scale=model_guidance_scale,
                generator=generator,
            )["sample"][0]
        except RuntimeError as e:
            raise RuntimeError(
                f"The model failed to generate a texture. Error message: {e}"
            )
            return

    else:
        if model_autocast and device != "cpu":
            with torch.autocast(device):
                try:
                    image = pipe(
                        prompt=texture_prompt,
                        num_inference_steps=model_steps,
                        guidance_scale=model_guidance_scale,
                        generator=generator,
                    )["sample"][0]
                except RuntimeError as e:
                    raise RuntimeError(
                        f"The model failed to generate a texture. Error message: {e}"
                    )
                    return
        else:
            try:
                image = pipe(
                    prompt=texture_prompt,
                    num_inference_steps=model_steps,
                    guidance_scale=model_guidance_scale,
                    generator=generator,
                )["sample"][0]
            except RuntimeError as e:
                raise RuntimeError(
                    f"The model failed to generate a texture. Error message: {e}"
                )
                return

    image_path = find_unique_filename(
        os.path.join(texture_path, texture_name) + texture_format
    )

    image.save(image_path)

    torch_gc()

    return image_path
