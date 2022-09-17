# SPDX-License-Identifier: GPL-2.0-or-later

import ctypes
import os
import solvent.constants as constants
import subprocess


def is_admin() -> bool:
    if constants.CURRENT_PLATFORM == "Windows":
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    else:
        return os.getuid() == 0


def install(package: constants.Package) -> None:
    if package.version is None:
        command = [
            constants.PYTHON_EXECUTABLE_LOCATION,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-warn-script-location",
            package.name,
        ]
    else:
        command = [
            constants.PYTHON_EXECUTABLE_LOCATION,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-warn-script-location",
            f"{package.name}=={package.version}",
        ]

    if package.extra_args is not None:
        command.extend(package.extra_args)

    subprocess.run(command, check=True, env=constants.ENVIRONMENT_VARIABLES)


def setup() -> None:
    # Ensure that pip is installed and up-to-date
    spec_output = subprocess.check_output(
        [
            constants.PYTHON_EXECUTABLE_LOCATION,
            "-c",
            "from importlib.util import find_spec; print(find_spec('pip'))",
        ],
        env=constants.ENVIRONMENT_VARIABLES,
    )
    if spec_output == b"None\n":
        subprocess.run(
            [constants.PYTHON_EXECUTABLE_LOCATION, "-m", "ensurepip"],
            check=True,
            env=constants.ENVIRONMENT_VARIABLES,
        )
    pip_upgrade_command = [
        constants.PYTHON_EXECUTABLE_LOCATION,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
    ]
    subprocess.run(
        pip_upgrade_command,
        check=True,
        env=constants.ENVIRONMENT_VARIABLES,
    )
