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
STABLE_DIFFUSION_DIRECTORY_NAME = "stable-diffusion-v1-4"
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
    version: str
    extra_args: Optional[List[str]] = None


@dataclass
class UserInput:
    texture_name: str
    texture_prompt: str
    texture_seed: int
    model_steps: int
    model_guidance_scale: float
    model_device: str
    texture_format: str
    texture_path: str


# This is our version of requirements.txt
PIP = Package(name="pip", version="22.2.2")
REQUIRED_PACKAGES = []
if CURRENT_PLATFORM == "Darwin":
    REQUIRED_PACKAGES.append(Package(name="torch", version="1.12.1"))
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
REQUIRED_PACKAGES.append(Package(name="diffusers", version="0.2.4"))
REQUIRED_PACKAGES.append(Package(name="transformers", version="4.21.3"))
REQUIRED_PACKAGES = tuple(REQUIRED_PACKAGES)
OPTIONAL_PACKAGES = (
    Package(name="ftfy", version="6.1.1"),
    Package(name="spacy", version="3.4.1"),
)
