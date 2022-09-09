# SPDX-License-Identifier: GPL-2.0-or-later

bl_info = {
    "name": "Solvent",
    "description": "Texture Generation Using Stable Diffusion Model in Blender",
    "author": "James Raphael Tiovalen",
    "version": (0, 0, 1),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Solvent",
    "warning": "This requires installation of additional packages and is currently in heavy development.",
    "support": "COMMUNITY",
    "doc_url": "https://github.com/jamestiotio/solvent",
    "tracker_url": "https://github.com/jamestiotio/solvent/issues",
    "category": "Development",
}

import bpy
import importlib
import secrets
import solvent.callbacks as callbacks
import solvent.constants as constants
import solvent.utils as utils
import subprocess
import tempfile
from typing import Set

# There is no way to check the current status of the Console Window (open/closed), so we assume that the user has not opened it yet
BLENDER_CONSOLE_WINDOW_OPENED = False
REQUIRED_PACKAGES_INSTALLED = False


def import_module(package: constants.Package) -> None:
    package_name = package.name
    if not package_name in globals():
        globals()[package_name] = importlib.import_module(package_name)


class SolventUserInput(bpy.types.PropertyGroup):
    texture_name: bpy.props.StringProperty(
        name="Texture Name",
        description="Name of the texture used for the output file",
    )
    texture_prompt: bpy.props.StringProperty(
        name="Texture Prompt",
        description="Prompt used for the texture",
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
        description="The number of denoising iterations. Larger values would take a longer time to finish but would generate higher-quality textures",
    )
    model_guidance_scale: bpy.props.FloatProperty(
        name="Model Guidance Scale",
        default=7.5,
        min=1,
        description="The adherence to the text prompt and sample quality. Higher values would generate more accurate, but less diverse, textures. A value of 1 would disable classifier-free guidance",
    )
    texture_tileable: bpy.props.BoolProperty(
        name="Tileable",
        default=True,
        description="Whether the generated texture should be tileable or not",
    )
    model_attention_slicing: bpy.props.BoolProperty(
        name="Attention Slicing",
        default=True,
        description="Whether to chunk the attention computation or not. Slicing the attention computation would reduce GPU VRAM usage but it would slightly increase the time taken to generate the texture. It's highly recommended to keep this enabled",
    )
    model_autocast: bpy.props.BoolProperty(
        name="Autocast",
        default=True,
        description="Whether to use automatic mixed precision or not. Mixed precision would take a shorter time to generate the texture but it would slightly reduce the quality of the texture. It's highly recommended to keep this enabled.\nIf you use half precision, autocast must be enabled.\nIf you use CPU, you must disable autocast",
        update=lambda self, context: callbacks.update_model_autocast(self, context),
    )
    if constants.CURRENT_PLATFORM == "Darwin":
        model_precision: bpy.props.EnumProperty(
            name="Model Precision",
            items=[
                ("Full", "Full", "Full Precision"),
            ],
            default="Full",
            description="The precision of the PyTorch model. Higher precision might generate higher-quality textures but would require more GPU VRAM",
        )
    else:
        model_precision: bpy.props.EnumProperty(
            name="Model Precision",
            items=[
                ("Half", "Half", "Half Precision"),
                ("Full", "Full", "Full Precision"),
            ],
            default="Half",
            description="The precision of the PyTorch model. Higher precision might generate higher-quality textures but would require more GPU VRAM. It's highly recommended to use half precision.\nIf you use half precision, autocast must be enabled.\nIf you use CPU, you must use full precision and disable autocast",
            update=lambda self, context: callbacks.update_model_precision_and_autocast(
                self, context
            ),
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
            description="The device used by the model to perform the texture generation.\nIf you use CPU, you must use full precision and disable autocast",
            update=lambda self, context: callbacks.update_model_precision_and_autocast(
                self, context
            ),
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
            description="The device used by the model to perform the texture generation.\nIf you use CPU, you must use full precision and disable autocast",
            update=lambda self, context: callbacks.update_model_precision_and_autocast(
                self, context
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
        maxlen=1024,
        default=tempfile.gettempdir(),
        description="Select path to export texture to",
    )


class SolventGenerateTexture(bpy.types.Operator):
    bl_idname = "solvent.generate_textures"
    bl_label = "Generate Texture"
    bl_description = "Generate texture using the Stable Diffusion model"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context) -> Set[str]:
        user_input = constants.UserInput(
            texture_name=bpy.context.scene.input_tool.texture_name,
            texture_prompt=bpy.context.scene.input_tool.texture_prompt,
            texture_seed=bpy.context.scene.input_tool.texture_seed,
            model_steps=bpy.context.scene.input_tool.model_steps,
            model_guidance_scale=bpy.context.scene.input_tool.model_guidance_scale,
            texture_tileable=bpy.context.scene.input_tool.texture_tileable,
            model_attention_slicing=bpy.context.scene.input_tool.model_attention_slicing,
            model_autocast=bpy.context.scene.input_tool.model_autocast,
            model_precision=bpy.context.scene.input_tool.model_precision,
            model_device=bpy.context.scene.input_tool.model_device,
            texture_format=bpy.context.scene.input_tool.texture_format,
            texture_path=bpy.path.abspath(bpy.context.scene.input_tool.texture_path),
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

        from solvent.model import text2image

        image_path = text2image(user_input)

        if image_path is None:
            self.report({"ERROR"}, "Texture generation failed!")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Texture has been generated successfully! Texture image is located at: {image_path}",
        )

        try:
            material = bpy.context.active_object.material_slots[0].material
            texture_image = bpy.data.images.load(image_path)
            material.use_nodes = True
            material_output = material.node_tree.nodes.get("Material Output")
            principled_bsdf = material.node_tree.nodes.get("Principled BSDF")
            texture_node = material.node_tree.nodes.new("ShaderNodeTexImage")
            texture_node.image = texture_image
            material.node_tree.links.new(
                texture_node.outputs[0], principled_bsdf.inputs[0]
            )

            self.report(
                {"INFO"}, "Texture has been applied to the currently active material!"
            )
        except Exception as e:
            self.report(
                {"WARNING"},
                f"Failed to apply texture to the material! Error message: {e}",
            )
            return {"CANCELLED"}

        return {"FINISHED"}

    def invoke(self, context, event) -> Set[str]:
        return context.window_manager.invoke_confirm(self, event)


class SolventMainPanel(bpy.types.Panel):
    bl_label = "Solvent"
    bl_idname = "SOLVENT_PT_Main"
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
        row.prop(input_tool, "texture_format")

        row = layout.row()
        row.prop(input_tool, "texture_path")

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
            if hasattr(globals()[package.name], "__version__"):
                layout.label(
                    text=f"{package.name} v{globals()[package.name].__version__}",
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
                    spec_output = subprocess.check_output(
                        [
                            constants.PYTHON_EXECUTABLE_LOCATION,
                            "-c",
                            f"from importlib.util import find_spec; print(find_spec('{package.name}'))",
                        ],
                        env=constants.ENVIRONMENT_VARIABLES,
                    )
                    if spec_output == b"None\n":
                        utils.install(package)
                    import_module(package)
                    if (
                        package.version is not None
                        and hasattr(globals()[package.name], "__version__")
                        and globals()[package.name].__version__ != package.version
                    ):
                        utils.install(package)
            except (subprocess.CalledProcessError, ImportError):
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
                    pip_install_commands += (
                        f"{preamble} {package.name}=={package.version} {extra_args}\n"
                    )
                pip_install_commands = pip_install_commands[:-1]
                env_var_command = (
                    "set PYTHONNOUSERSITE=1"
                    if constants.CURRENT_PLATFORM == "Windows"
                    else "export PYTHONNOUSERSITE=1"
                )
                error_message = (
                    "Attempt to automatically install the required Python packages has failed. "
                    "Please install the Python packages manually by running the following commands in a terminal:\n"
                    f"{env_var_command}\n"
                    f"{pip_install_commands}"
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
                type=SolventUserInput
            )

            return {"FINISHED"}

        else:
            permission_level = (
                "elevated permissions"
                if constants.CURRENT_PLATFORM == "Windows"
                else "non-elevated permissions"
            )
            error_message = f"Blender must be started with {permission_level} in order to install the required Python packages properly."
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
            f"Please install the necessary packages for the \"{bl_info.get('name')}\" add-on.",
            f"1. Go to Edit > Preferences > Add-ons.",
            f"2. Search for the add-on with the name \"{bl_info.get('category')}: {bl_info.get('name')}\"",
            f"    under the {bl_info.get('support').title()} tab and enable it.",
            f'3. Under "Preferences", click on the "{SolventInstallPackages.bl_label}"',
            f"    button. This will download and install the required packages,",
            f"    if Blender has the required permissions. If you are experiencing issues:",
            f"1. If you are on Windows, re-open Blender with administrator privileges.",
            f"2. Otherwise, try installing Blender from Blender's official site",
            f"    (https://www.blender.org/download/) and extract it to a",
            f"    directory that you own the permissions to access.",
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
    SolventUserInput,
    SolventGenerateTexture,
    SolventMainPanel,
    SolventAboutPanel,
)

register, unregister = bpy.utils.register_classes_factory(classes)


def register() -> None:
    if bpy.app.version < bl_info["blender"]:
        current_blender_version = ".".join(str(i) for i in bpy.app.version)
        required_blender_version = ".".join(str(i) for i in bl_info["blender"])
        raise Exception(
            f"This add-on doesn't support Blender version less than {required_blender_version}. "
            f"Blender version {required_blender_version} or greater is recommended, "
            f"but the current version is {current_blender_version}."
        )

    for cls in preparation_classes:
        bpy.utils.register_class(cls)

    try:
        for package in constants.REQUIRED_PACKAGES:
            import_module(package)

        # Check currently-imported packages for version compatibility
        for package in constants.REQUIRED_PACKAGES:
            if (
                package.version is not None
                and hasattr(globals()[package.name], "__version__")
                and globals()[package.name].__version__ != package.version
            ):
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
        bpy.types.Scene.input_tool = bpy.props.PointerProperty(type=SolventUserInput)


def unregister() -> None:
    for cls in preparation_classes:
        bpy.utils.unregister_class(cls)

    if REQUIRED_PACKAGES_INSTALLED:
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)

        del bpy.types.Scene.input_tool


if __name__ == "__main__":
    register()
