# SPDX-License-Identifier: GPL-2.0-or-later

from diffusers import StableDiffusionPipeline
import os
import solvent.constants as constants
from typing import Optional
import torch


def torch_gc():
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
    model_device = user_input.model_device
    texture_format = user_input.texture_format
    texture_path = user_input.texture_path

    device = "cuda" if model_device == "GPU" else "cpu"

    # Overwrite the device value if CUDA is not available
    # This might go against the user's specified settings, but it's better than just throwing an error
    if not torch.cuda.is_available():
        device = "cpu"

    torch_gc()

    generator = torch.Generator(device).manual_seed(texture_seed)

    pipe = StableDiffusionPipeline.from_pretrained(
        constants.MODEL_PATH,
        # TODO: Remove the following two lines when this PR is merged and included in a release version: https://github.com/huggingface/diffusers/pull/366
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(device)

    with torch.autocast(device):
        try:
            image = pipe(
                prompt=texture_prompt,
                num_inference_steps=model_steps,
                guidance_scale=model_guidance_scale,
                generator=generator,
            )["sample"][0]
        except RuntimeError:
            return

    image_path = find_unique_filename(
        os.path.join(texture_path, texture_name) + texture_format
    )

    image.save(image_path)

    return image_path
