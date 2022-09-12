# SPDX-License-Identifier: GPL-2.0-or-later


import bpy
import solvent.constants as constants


def update_model_autocast(self, context) -> None:
    # Autocast is not supported on CPU: https://huggingface.co/CompVis/stable-diffusion-v1-4/discussions/42
    if (
        bpy.context.scene.input_tool.model_device == "CPU"
        and bpy.context.scene.input_tool.model_autocast
    ):
        bpy.context.scene.input_tool.model_autocast = False
    # Half precision without autocast is not supported: https://github.com/huggingface/diffusers/issues/234
    elif (
        bpy.context.scene.input_tool.model_precision == "Half"
        and not bpy.context.scene.input_tool.model_autocast
    ):
        bpy.context.scene.input_tool.model_autocast = True


def update_model_precision(self, context) -> None:
    # Half precision on CPU is not supported: https://github.com/huggingface/transformers/issues/16378
    if (
        bpy.context.scene.input_tool.model_device == "CPU"
        and bpy.context.scene.input_tool.model_precision == "Half"
    ):
        bpy.context.scene.input_tool.model_precision = "Full"


def update_model_precision_and_autocast(self, context) -> None:
    update_model_precision(self, context)
    update_model_autocast(self, context)


def update_batching(self, context) -> None:
    # Batching is not reliable on Apple MPS: https://github.com/huggingface/diffusers/issues/363
    if bpy.context.scene.input_tool.batching and constants.CURRENT_PLATFORM == "Darwin":
        bpy.context.scene.input_tool.batching = False
