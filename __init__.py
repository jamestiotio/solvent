# SPDX-License-Identifier: GPL-2.0-or-later

bl_info = {
    "name": "Solvent",
    "description": "AI-Assisted Texture Generation Toolkit in Blender",
    "author": "James Raphael Tiovalen",
    "version": (0, 0, 1),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Solvent",
    "warning": (
        "This requires installation of additional packages and is currently in heavy"
        " development."
    ),
    "support": "COMMUNITY",
    "doc_url": "https://github.com/jamestiotio/solvent",
    "tracker_url": "https://github.com/jamestiotio/solvent/issues",
    "category": "Development",
}

import importlib
import os
import secrets
import subprocess
import tempfile
from typing import Set

import bpy

import solvent.callbacks as callbacks
import solvent.constants as constants
import solvent.utils as utils

# There is no way to check the current status of the Console Window (open/closed), so we assume that the user has not opened it yet
BLENDER_CONSOLE_WINDOW_OPENED = False
REQUIRED_PACKAGES_INSTALLED = False


def import_module(package: constants.Package) -> None:
    package_name = package.import_name if package.import_name else package.name
    if not package_name in globals():
        globals()[package_name] = importlib.import_module(package_name)


class SolventTextureGenerationUserInput(bpy.types.PropertyGroup):
    texture_name: bpy.props.StringProperty(
        name="Texture Name",
        description="Name of the texture used for the output file",
    )
    texture_prompt: bpy.props.StringProperty(
        name="Texture Prompt",
        description="Prompt used for the texture",
    )
    texture_initial_image: bpy.props.StringProperty(
        name="Initial Image",
        subtype="FILE_PATH",
        maxlen=259,
        description="Select path to the initial image, if any",
    )
    texture_mask_image: bpy.props.StringProperty(
        name="Mask Image",
        subtype="FILE_PATH",
        maxlen=259,
        description="Select path to the mask image, if any",
    )
    texture_variation_strength: bpy.props.FloatProperty(
        name="Variation Strength",
        default=0.8,
        min=0.0,
        max=1.0,
        description=(
            "How much to transform the initial image, if any. A value of 1 would ignore"
            " the initial image"
        ),
    )
    texture_seed: bpy.props.IntProperty(
        name="Texture Seed",
        default=secrets.choice(range(2**31)),
        min=0,
        description="The seed used by the PyTorch generator",
    )
    model_steps: bpy.props.IntProperty(
        name="Model Steps",
        default=50,
        min=1,
        description=(
            "The number of denoising iterations. Larger values would take a longer time"
            " to finish but would generate higher-quality textures"
        ),
    )
    model_guidance_scale: bpy.props.FloatProperty(
        name="Model Guidance Scale",
        default=7.5,
        min=1,
        description=(
            "The adherence to the text prompt and sample quality. Higher values would"
            " generate more accurate, but less diverse, textures. A value of 1 would"
            " disable classifier-free guidance"
        ),
    )
    texture_tileable: bpy.props.BoolProperty(
        name="Tileable",
        default=True,
        description="Whether the generated texture should be tileable or not",
    )
    model_attention_slicing: bpy.props.BoolProperty(
        name="Attention Slicing",
        default=True,
        description=(
            "Whether to chunk the attention computation or not. Slicing the attention"
            " computation would reduce GPU VRAM usage but it would slightly increase"
            " the time taken to generate the texture. It's highly recommended to keep"
            " this enabled"
        ),
    )
    model_autocast: bpy.props.BoolProperty(
        name="Autocast",
        default=True,
        description=(
            "Whether to use automatic mixed precision or not. Mixed precision would"
            " take a shorter time to generate the texture but it would slightly reduce"
            " the quality of the texture. It's highly recommended to keep this"
            " enabled.\n\nIf you use half precision, autocast must be enabled.\n\nIf"
            " you use CPU, you must disable autocast"
        ),
        update=callbacks.update_model_autocast,
    )
    if constants.CURRENT_PLATFORM == "Darwin":
        model_precision: bpy.props.EnumProperty(
            name="Model Precision",
            items=[
                ("Full", "Full", "Full Precision"),
            ],
            default="Full",
            description=(
                "The precision of the PyTorch model. Higher precision might generate"
                " higher-quality textures but would require more GPU VRAM"
            ),
        )
    else:
        model_precision: bpy.props.EnumProperty(
            name="Model Precision",
            items=[
                ("Half", "Half", "Half Precision"),
                ("Full", "Full", "Full Precision"),
            ],
            default="Half",
            description=(
                "The precision of the PyTorch model. Higher precision might generate"
                " higher-quality textures but would require more GPU VRAM. It's highly"
                " recommended to use half precision.\n\nIf you use half precision,"
                " autocast must be enabled.\n\nIf you use CPU, you must use full"
                " precision and disable autocast"
            ),
            update=callbacks.update_model_precision_and_autocast,
        )
    if constants.CURRENT_PLATFORM == "Darwin":
        # Assume that the user uses an M1 Mac
        # Should use torch.backends.mps.is_available() to display available options but it requires the PyTorch package to be installed in the first place (cyclic dependency problem)
        model_device: bpy.props.EnumProperty(
            name="Model Device",
            items=[
                ("MPS", "MPS", "Use Apple's MPS (generally faster)"),
                ("CPU", "CPU", "Use CPU (generally slower)"),
            ],
            default="MPS",
            description=(
                "The device used by the model to perform the texture generation.\n\nIf"
                " you use CPU, you must use full precision and disable autocast"
            ),
            update=callbacks.update_model_precision_and_autocast,
        )
    else:
        # Should use torch.cuda.is_available() to display available options but it requires the PyTorch package to be installed in the first place (cyclic dependency problem)
        model_device: bpy.props.EnumProperty(
            name="Model Device",
            items=[
                ("GPU", "GPU", "Use GPU (generally faster)"),
                ("CPU", "CPU", "Use CPU (generally slower)"),
            ],
            default="GPU",
            description=(
                "The device used by the model to perform the texture generation.\n\nIf"
                " you use CPU, you must use full precision and disable autocast"
            ),
            update=callbacks.update_model_precision_and_autocast,
        )
    model_scheduler: bpy.props.EnumProperty(
        name="Model Scheduler",
        items=[
            ("DDIM", "DDIM", "Use DDIM Scheduler"),
            ("K-LMS", "K-LMS", "Use K-LMS Scheduler"),
            ("PNDM", "PNDM", "Use PNDM Scheduler"),
        ],
        default="PNDM",
        description=(
            "The scheduler used by the model to perform the texture generation. K-LMS"
            " would provide a good balance between quality and speed, while DDIM is"
            " generally faster"
        ),
    )
    texture_format: bpy.props.EnumProperty(
        name="Texture Format",
        items=[
            (".png", ".png", "Export texture as a PNG file"),
            (".jpg", ".jpg", "Export texture as a JPG file"),
        ],
        default=".png",
        description="Select texture file format",
    )
    texture_path: bpy.props.StringProperty(
        name="Texture Path",
        subtype="DIR_PATH",
        maxlen=259,
        default=tempfile.gettempdir(),
        description="Select path to export texture to",
    )
    num_of_images: bpy.props.IntProperty(
        name="Number of Images",
        default=1,
        min=1,
        max=4,
        description=(
            "The number of texture images to generate. If batching is disabled, more"
            " textures generated would increase the time taken to generate all of the"
            " textures. Otherwise, more textures generated would increase the GPU VRAM"
            " required"
        ),
    )
    batching: bpy.props.BoolProperty(
        name="Batching",
        default=False,
        description=(
            "Whether to batch the texture generation or not. Batching would reduce the"
            " time taken to generate multiple textures but it would increase the GPU"
            " VRAM required. It's highly recommended to keep this disabled due to"
            " limited memory constraints in consumer-grade hardware.\n\nIf you use an"
            " M1 Mac, you must disable batching"
        ),
        update=callbacks.update_batching,
    )


class SolventGenerateTexture(bpy.types.Operator):
    bl_idname = "solvent.generate_textures"
    bl_label = "Generate Texture"
    bl_description = "Generate texture using the Stable Diffusion model"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context) -> Set[str]:
        user_input = constants.TextureGenerationUserInput(
            texture_name=bpy.context.scene.input_tool.texture_name,
            texture_prompt=bpy.context.scene.input_tool.texture_prompt,
            texture_initial_image=bpy.path.abspath(
                bpy.context.scene.input_tool.texture_initial_image
            ),
            texture_mask_image=bpy.path.abspath(
                bpy.context.scene.input_tool.texture_mask_image
            ),
            texture_variation_strength=bpy.context.scene.input_tool.texture_variation_strength,
            texture_seed=bpy.context.scene.input_tool.texture_seed,
            model_steps=bpy.context.scene.input_tool.model_steps,
            model_guidance_scale=bpy.context.scene.input_tool.model_guidance_scale,
            texture_tileable=bpy.context.scene.input_tool.texture_tileable,
            model_attention_slicing=bpy.context.scene.input_tool.model_attention_slicing,
            model_autocast=bpy.context.scene.input_tool.model_autocast,
            model_precision=bpy.context.scene.input_tool.model_precision,
            model_device=bpy.context.scene.input_tool.model_device,
            model_scheduler=bpy.context.scene.input_tool.model_scheduler,
            texture_format=bpy.context.scene.input_tool.texture_format,
            texture_path=bpy.path.abspath(bpy.context.scene.input_tool.texture_path),
            num_of_images=bpy.context.scene.input_tool.num_of_images,
            batching=bpy.context.scene.input_tool.batching,
        )

        if not user_input.texture_name:
            self.report({"ERROR"}, "Please specify a texture name.")
            return {"CANCELLED"}

        if not user_input.texture_prompt:
            self.report({"ERROR"}, "Please specify a texture prompt.")
            return {"CANCELLED"}

        global BLENDER_CONSOLE_WINDOW_OPENED

        if (
            constants.CURRENT_PLATFORM == "Windows"
            and not BLENDER_CONSOLE_WINDOW_OPENED
        ):
            bpy.ops.wm.console_toggle()
            BLENDER_CONSOLE_WINDOW_OPENED = True

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        try:
            from solvent.model import generate_texture

            image_paths = generate_texture(user_input)
        except Exception as e:
            self.report({"ERROR"}, f"Something went wrong! Error message: {e}")
            return {"CANCELLED"}

        if not image_paths:
            self.report({"ERROR"}, "Texture generation failed!")
            return {"CANCELLED"}

        if len(image_paths) == 1:
            self.report(
                {"INFO"},
                "Texture has been generated successfully! Texture image is located at:"
                f" {image_paths[0]}",
            )
        else:
            self.report(
                {"INFO"},
                f"Textures have been generated successfully!",
            )

        try:
            for idx, image_path in enumerate(image_paths):
                if idx == 0:
                    texture_image = bpy.data.images.load(image_path)
                else:
                    bpy.data.images.load(image_path)

            material = bpy.context.active_object.material_slots[0].material
            material.use_nodes = True
            material_output = material.node_tree.nodes.get("Material Output")
            principled_bsdf = material.node_tree.nodes.get("Principled BSDF")
            texture_node = material.node_tree.nodes.new("ShaderNodeTexImage")
            texture_node.image = texture_image
            material.node_tree.links.new(
                texture_node.outputs[0], principled_bsdf.inputs[0]
            )

            if len(image_paths) == 1:
                self.report(
                    {"INFO"},
                    "Texture has been applied to the currently active material!",
                )
            else:
                self.report(
                    {"INFO"},
                    "The first texture has been applied to the currently active"
                    " material!",
                )
        except Exception as e:
            self.report(
                {"WARNING"},
                f"Failed to apply texture to the material! Error message: {e}",
            )

        return {"FINISHED"}

    def invoke(self, context, event) -> Set[str]:
        return context.window_manager.invoke_confirm(self, event)


class SolventTexturePanel(bpy.types.Panel):
    bl_label = "Texture Generation"
    bl_idname = "SOLVENT_PT_Texture"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Solvent"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        scene = context.scene
        input_tool = scene.input_tool

        row = layout.row()
        row.prop(input_tool, "texture_name")

        row = layout.row()
        row.prop(input_tool, "texture_prompt")

        row = layout.row()
        row.prop(input_tool, "texture_initial_image")

        row = layout.row()
        row.prop(input_tool, "texture_mask_image")

        row = layout.row()
        row.prop(input_tool, "texture_variation_strength")

        row = layout.row()
        row.prop(input_tool, "texture_seed")

        row = layout.row()
        row.prop(input_tool, "model_steps")

        row = layout.row()
        row.prop(input_tool, "model_guidance_scale")

        row = layout.row()
        row.prop(input_tool, "texture_tileable")
        row.prop(input_tool, "model_attention_slicing")
        row.prop(input_tool, "model_autocast")

        row = layout.row()
        row.prop(input_tool, "model_precision")

        row = layout.row()
        row.prop(input_tool, "model_device")

        row = layout.row()
        row.prop(input_tool, "model_scheduler")

        row = layout.row()
        row.prop(input_tool, "texture_format")

        row = layout.row()
        row.prop(input_tool, "texture_path")

        row = layout.row()
        row.prop(input_tool, "num_of_images")

        row = layout.row()
        row.prop(input_tool, "batching")

        layout.separator()

        self.layout.operator(
            "solvent.generate_textures",
            icon="DISCLOSURE_TRI_RIGHT",
            text="Generate Texture",
        )

        layout.separator()


class SolventAboutPanel(bpy.types.Panel):
    bl_label = "About"
    bl_idname = "SOLVENT_PT_About"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Solvent"

    def draw(self, context) -> None:
        layout = self.layout
        scene = context.scene
        input_tool = scene.input_tool

        current_addon_version = ".".join(str(i) for i in bl_info["version"])
        layout.label(text=f"Solvent v{current_addon_version}")

        layout.label(
            text=f"Stable Diffusion v{constants.CURRENT_SD_VERSION}",
            icon="EXPERIMENTAL",
        )

        layout.label(text="Installed Packages:")
        for package in constants.REQUIRED_PACKAGES:
            package_name = package.import_name if package.import_name else package.name
            if hasattr(globals()[package_name], "__version__"):
                layout.label(
                    text=f"{package.name} v{globals()[package_name].__version__}",
                    icon="PACKAGE",
                )
            else:
                layout.label(text=f"{package.name}", icon="PACKAGE")


class SolventInstallPackages(bpy.types.Operator):
    bl_idname = "solvent.install_packages"
    bl_label = "Install Python packages"
    bl_description = (
        "Download and install the required additional Python packages for this add-on. "
        "Internet connection is required and Blender may have to be started with "
        "elevated permissions in order to install the packages"
    )
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(cls, context) -> bool:
        return not REQUIRED_PACKAGES_INSTALLED

    def execute(self, context) -> Set[str]:
        if (constants.CURRENT_PLATFORM == "Windows" and utils.is_admin()) or (
            constants.CURRENT_PLATFORM != "Windows" and not utils.is_admin()
        ):
            try:
                global BLENDER_CONSOLE_WINDOW_OPENED
                if (
                    constants.CURRENT_PLATFORM == "Windows"
                    and not BLENDER_CONSOLE_WINDOW_OPENED
                ):
                    bpy.ops.wm.console_toggle()
                    BLENDER_CONSOLE_WINDOW_OPENED = True
                utils.setup()
                for package in constants.REQUIRED_PACKAGES:
                    package_name = (
                        package.import_name if package.import_name else package.name
                    )
                    spec_output = subprocess.check_output(
                        [
                            constants.PYTHON_EXECUTABLE_LOCATION,
                            "-c",
                            (
                                "from importlib.util import find_spec;"
                                f" print(find_spec('{package_name}'))"
                            ),
                        ],
                        env=constants.ENVIRONMENT_VARIABLES,
                    )
                    if spec_output in [b"None\n", b"None\r\n"]:
                        utils.install(package)
                    import_module(package)
                    if package.version is not None and hasattr(
                        globals()[package_name], "__version__"
                    ):
                        installed_package_version = globals()[
                            package_name
                        ].__version__.split(".")
                        installed_package_version += ["0"] * (
                            3 - len(installed_package_version)
                        )
                        for idx, val in enumerate(installed_package_version):
                            installed_package_version[idx] = int(val.split("+")[0])
                        installed_package_version = tuple(
                            installed_package_version[0:3]
                        )
                        target_package_version = package.version.split(".")
                        target_package_version += ["0"] * (
                            3 - len(target_package_version)
                        )
                        for idx, val in enumerate(target_package_version):
                            target_package_version[idx] = int(val.split("+")[0])
                        target_package_version = tuple(target_package_version[0:3])
                        if installed_package_version < target_package_version:
                            utils.install(package)
            except (subprocess.CalledProcessError, ImportError) as e:
                pip_install_commands = ""
                preamble = (
                    f'"{constants.PYTHON_EXECUTABLE_LOCATION}" -m pip install --upgrade'
                )
                for package in constants.REQUIRED_PACKAGES:
                    extra_args = (
                        " ".join(package.extra_args)
                        if package.extra_args is not None
                        else ""
                    )
                    if package.version is not None:
                        pip_install_commands += f"{preamble} {package.name}=={package.version} {extra_args}\n"
                    else:
                        pip_install_commands += (
                            f"{preamble} {package.name} {extra_args}\n"
                        )
                pip_install_commands = pip_install_commands[:-1]
                env_var_command = (
                    "set PYTHONNOUSERSITE=1"
                    if constants.CURRENT_PLATFORM == "Windows"
                    else "export PYTHONNOUSERSITE=1"
                )
                error_message = (
                    "Attempt to automatically install the required Python packages has"
                    " failed. Please install the Python packages manually by running"
                    " the following commands in a"
                    f" terminal:\n{env_var_command}\n{pip_install_commands}"
                )
                self.report({"ERROR"}, str(error_message))

                return {"CANCELLED"}

            except Exception as e:
                self.report({"ERROR"}, f"Something went wrong! Error message: {e}")
                return {"CANCELLED"}

            global REQUIRED_PACKAGES_INSTALLED
            REQUIRED_PACKAGES_INSTALLED = True

            for cls in classes:
                bpy.utils.register_class(cls)

            bpy.types.Scene.input_tool = bpy.props.PointerProperty(
                type=SolventTextureGenerationUserInput
            )

            return {"FINISHED"}

        else:
            permission_level = (
                "elevated permissions"
                if constants.CURRENT_PLATFORM == "Windows"
                else "non-elevated permissions"
            )
            error_message = (
                f"Blender must be started with {permission_level} in order to install"
                " the required Python packages properly."
            )
            self.report({"ERROR"}, str(error_message))
            return {"CANCELLED"}


class SolventWarningPanel(bpy.types.Panel):
    bl_label = "Warning"
    bl_idname = "SOLVENT_PT_Warning"
    bl_category = "Solvent"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context) -> bool:
        return not REQUIRED_PACKAGES_INSTALLED

    def draw(self, context) -> None:
        layout = self.layout

        lines = [
            (
                "Please install the necessary packages for the"
                f" \"{bl_info.get('name')}\" add-on."
            ),
            f"1. Go to Edit > Preferences > Add-ons.",
            (
                f"2. Search for the add-on with the name \"{bl_info.get('category')}:"
                f" {bl_info.get('name')}\""
            ),
            f"    under the {bl_info.get('support').title()} tab and enable it.",
            f'3. Under "Preferences", click on the "{SolventInstallPackages.bl_label}"',
            f"    button. This will download and install the required packages,",
            f"    if Blender has the required permissions.",
            (
                f"4. After the installation has finished, you would need to restart"
                f" Blender."
            ),
            f"",
            f"If you are experiencing issues:",
            f"1. If you are on Windows, re-open Blender with administrator privileges.",
            f"2. Otherwise, try installing Blender from Blender's official site and",
            f"    extract it to a directory that you own the permissions to access.",
        ]

        for line in lines:
            layout.label(text=line)


class SolventPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context) -> None:
        layout = self.layout
        layout.operator(SolventInstallPackages.bl_idname, icon="CONSOLE")


preparation_classes = (
    SolventWarningPanel,
    SolventInstallPackages,
    SolventPreferences,
)
classes = (
    SolventTextureGenerationUserInput,
    SolventGenerateTexture,
    SolventTexturePanel,
    SolventAboutPanel,
)

register, unregister = bpy.utils.register_classes_factory(classes)


def register() -> None:
    if bpy.app.version < bl_info["blender"]:
        current_blender_version = ".".join(str(i) for i in bpy.app.version)
        required_blender_version = ".".join(str(i) for i in bl_info["blender"])
        raise Exception(
            "This add-on doesn't support Blender version less than"
            f" {required_blender_version}. Blender version"
            f" {required_blender_version} or greater is recommended, but the current"
            f" version is {current_blender_version}."
        )

    for cls in preparation_classes:
        bpy.utils.register_class(cls)

    try:
        for package in constants.REQUIRED_PACKAGES:
            import_module(package)

        # Check currently-imported packages for version compatibility
        for package in constants.REQUIRED_PACKAGES:
            package_name = package.import_name if package.import_name else package.name
            if package.version is not None and hasattr(
                globals()[package_name], "__version__"
            ):
                installed_package_version = globals()[package_name].__version__.split(
                    "."
                )
                installed_package_version += ["0"] * (
                    3 - len(installed_package_version)
                )
                for idx, val in enumerate(installed_package_version):
                    installed_package_version[idx] = int(val.split("+")[0])
                installed_package_version = tuple(installed_package_version[0:3])
                target_package_version = package.version.split(".")
                target_package_version += ["0"] * (3 - len(target_package_version))
                for idx, val in enumerate(target_package_version):
                    target_package_version[idx] = int(val.split("+")[0])
                target_package_version = tuple(target_package_version[0:3])
                if installed_package_version < target_package_version:
                    # Let the user manually update the packages at this point
                    return
        global REQUIRED_PACKAGES_INSTALLED
        REQUIRED_PACKAGES_INSTALLED = True
    except ModuleNotFoundError:
        # Let the user manually installs the necessary packages at this point
        return
    except Exception as e:
        raise Exception(f"Something went wrong! Error message: {e}")

    for cls in classes:
        bpy.utils.register_class(cls)

    if REQUIRED_PACKAGES_INSTALLED:
        bpy.types.Scene.input_tool = bpy.props.PointerProperty(
            type=SolventTextureGenerationUserInput
        )


def unregister() -> None:
    for cls in preparation_classes:
        bpy.utils.unregister_class(cls)

    if REQUIRED_PACKAGES_INSTALLED:
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)

        del bpy.types.Scene.input_tool


if __name__ == "__main__":
    register()
