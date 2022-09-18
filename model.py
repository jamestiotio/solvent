# SPDX-License-Identifier: GPL-2.0-or-later

from contextlib import nullcontext
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
import gc
import os
from PIL import Image
import solvent.constants as constants
from typing import Optional, List
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


def generate_texture(
    user_input: constants.TextureGenerationUserInput,
) -> Optional[List]:
    texture_name = user_input.texture_name
    texture_prompt = user_input.texture_prompt
    texture_initial_image = user_input.texture_initial_image
    texture_mask_image = user_input.texture_mask_image
    texture_variation_strength = user_input.texture_variation_strength
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
    num_of_images = user_input.num_of_images
    batching = user_input.batching

    if batching:
        texture_prompt = [texture_prompt] * num_of_images

    # Overwrite the device value if CUDA/MPS is not available
    # This might go against the user's specified settings, but it's better than just throwing an error
    if model_device in ["GPU", "MPS"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_device == "MPS":
        device = "mps" if torch.backends.mps.is_available() else device
    if model_device == "CPU":
        device = "cpu"

    if model_scheduler == "DDIM":
        scheduler = ddim_scheduler
    elif model_scheduler == "K-LMS":
        scheduler = lms_scheduler
    else:
        # By default, use the PNDM scheduler
        scheduler = pndm_scheduler

    torch_gc()

    generator = torch.Generator(device).manual_seed(texture_seed)

    # Conditionally build the parameters
    if (
        constants.CURRENT_PLATFORM != "Darwin"
        and model_precision == "Half"
        and model_autocast
        and device != "cpu"
    ):
        pipeline_params = {
            "pretrained_model_name_or_path": constants.MODEL_PATH,
            "local_files_only": True,
            "use_auth_token": False,
            "revision": "fp16",
            "torch_dtype": torch.float16,
            "scheduler": scheduler,
        }
    else:
        pipeline_params = {
            "pretrained_model_name_or_path": constants.MODEL_PATH,
            "local_files_only": True,
            "use_auth_token": False,
            "scheduler": scheduler,
        }

    if texture_initial_image:
        if texture_mask_image:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(**pipeline_params).to(
                device
            )
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(**pipeline_params).to(
                device
            )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(**pipeline_params).to(device)

    pipe.set_progress_bar_config(disable=False, desc="Generating texture")

    if model_attention_slicing:
        pipe.enable_attention_slicing()

    if texture_tileable:
        # Only patch the `Conv2d` modules in the components that have it instead of globally (more efficient and better granularity of control)
        targets = [pipe.vae, pipe.unet]
        for target in targets:
            for module in target.modules():
                if isinstance(module, torch.nn.Conv2d):
                    # Patch to make tileable textures: https://gitlab.com/-/snippets/2395088
                    module.padding_mode = "circular"

    if texture_initial_image:
        with open(texture_initial_image, "rb") as initial_image_file:
            initial_image = (
                Image.open(initial_image_file).convert("RGB").resize((512, 512))
            )
        if texture_mask_image:
            with open(texture_mask_image, "rb") as mask_image_file:
                mask_image = (
                    Image.open(mask_image_file).convert("RGB").resize((512, 512))
                )

    images = []

    pipe_args = {
        "prompt": texture_prompt,
        "num_inference_steps": model_steps,
        "guidance_scale": model_guidance_scale,
        "generator": generator,
    }

    # Conditionally build the arguments
    if texture_initial_image:
        pipe_args["init_image"] = initial_image
        pipe_args["strength"] = texture_variation_strength
        if texture_mask_image:
            pipe_args["mask_image"] = mask_image

    # Conditionally select the suitable context manager
    if model_autocast and device != "cpu" and constants.CURRENT_PLATFORM != "Darwin":
        use_autocast = True
    else:
        use_autocast = False

    try:
        if constants.CURRENT_PLATFORM == "Darwin":
            # TODO: First-time "warmup" pass (https://github.com/huggingface/diffusers/issues/372)
            _ = pipe(texture_prompt, num_inference_steps=1)

        if batching:
            with torch.autocast(device) if use_autocast else nullcontext():
                images = pipe(**pipe_args).images
        else:
            for _ in range(num_of_images):
                with torch.autocast(device) if use_autocast else nullcontext():
                    image = pipe(**pipe_args).images[0]
                    images.append(image)

    except RuntimeError as e:
        raise RuntimeError(
            f"The model failed to generate a texture. Error message: {e}"
        )

    image_paths = []

    for image in images:
        image_path = find_unique_filename(
            os.path.join(texture_path, texture_name) + texture_format
        )
        image.save(image_path)
        image_paths.append(image_path)

    torch_gc()

    return image_paths
