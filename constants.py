# SPDX-License-Identifier: GPL-2.0-or-later

import bpy
from dataclasses import dataclass
import os
import platform
import sys
from typing import Optional, List

CURRENT_PLATFORM = platform.system()
CURRENT_SD_VERSION = "1.4"
PYTHON_EXECUTABLE_LOCATION = str(sys.executable)
STABLE_DIFFUSION_DIRECTORY_NAME = "models/stable-diffusion-v1-4"
MODEL_PATH = os.path.join(
    bpy.utils.user_resource("SCRIPTS"),
    "addons",
    "solvent",
    STABLE_DIFFUSION_DIRECTORY_NAME,
)
ENVIRONMENT_VARIABLES = os.environ
ENVIRONMENT_VARIABLES["PYTHONNOUSERSITE"] = "1"


@dataclass
class Package:
    name: str
    version: Optional[str] = None
    # Some packages might need extra arguments to be passed to pip to be installed properly
    extra_args: Optional[List[str]] = None
    # Some packages might have a different import name than their PyPI package name
    import_name: Optional[str] = None


@dataclass
class TextureGenerationUserInput:
    texture_name: str
    texture_prompt: str
    texture_initial_image: str
    texture_mask_image: str
    texture_variation_strength: float
    texture_seed: int
    model_steps: int
    model_guidance_scale: float
    texture_tileable: bool
    model_attention_slicing: bool
    model_autocast: bool
    model_precision: str
    model_device: str
    model_scheduler: str
    texture_format: str
    texture_path: str
    num_of_images: int
    batching: bool


# This is our version of requirements.txt
PIP = Package(name="pip", version="22.2.2")
REQUIRED_PACKAGES = []
if CURRENT_PLATFORM == "Darwin":
    REQUIRED_PACKAGES.append(
        # TODO: Update to stable version of PyTorch once it supports MPS
        Package(
            name="torch",
            extra_args=[
                "--pre",
                "-f",
                "https://download.pytorch.org/whl/nightly/torch/",
            ],
        )
    )
else:
    REQUIRED_PACKAGES.append(
        Package(
            name="torch",
            version="1.12.1+cu116",
            extra_args=[
                "-f",
                "https://download.pytorch.org/whl/torch_stable.html",
            ],
        )
    )
REQUIRED_PACKAGES.append(Package(name="diffusers", version="0.3.0"))
REQUIRED_PACKAGES.append(Package(name="transformers", version="4.21.3"))
REQUIRED_PACKAGES.append(Package(name="scipy", version="1.9.1"))
REQUIRED_PACKAGES.append(Package(name="ftfy", version="6.1.1"))
REQUIRED_PACKAGES.append(Package(name="spacy", version="3.4.1"))
REQUIRED_PACKAGES = tuple(REQUIRED_PACKAGES)
