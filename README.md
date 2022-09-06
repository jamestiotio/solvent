# Solvent
Texture Generation Using Stable Diffusion Model in Blender

## Prerequisites

Ensure that you already have the following software installed, preferably the latest versions:

- [Blender](https://www.blender.org/)
- [Git](https://git-scm.com/)
- [Git LFS](https://git-lfs.github.com/)

Additionally, you would also need to have a [Hugging Face](https://huggingface.co/) account to download the pre-trained Stable Diffusion model checkpoint weights. Optionally, you might want to create a [GitHub](https://github.com/) account as well to raise issues or pull requests to this repository.

## Instructions

> Note that if you are on Windows, you would need to run Blender (and hence, the add-on) as administrator with Internet connection in order to install all of the necessary Python packages. This only needs to be done once.
>
> Otherwise, you would need to install Blender by visiting the official site [here](https://www.blender.org/download/) instead of using other methods (i.e., such as using a package manager or `snap`):
>
> - Repositories used by package managers might not have the latest version of Blender.
> - More details about the issues of a `snap`-based installation can be found [here](https://developer.blender.org/T83085).
>
> Do also ensure that you possess the appropriate permissions to access the directory that you install Blender to.

1. Download the latest version of the add-on files from this repository:

   ```bash
   git clone --recurse-submodules https://github.com/jamestiotio/solvent
   ```

   Note that the size can be quite large as it includes the pre-trained Stable Diffusion model checkpoint weights. Depending on your Internet speed, you can go get a cup of coffee while waiting.

2. Zip up the `solvent` folder that has been cloned. This would create a `solvent.zip` file.

3. Install the add-on by opening Blender and following the instructions on the official Blender manual [here](https://docs.blender.org/manual/en/latest/editors/preferences/addons.html#installing-add-ons). Follow along the instructions to also enable the add-on.

4. Since the add-on will need to download and install additional Python packages, you can trigger the installation manually by going to `Edit > Preferences > Add-ons` and click the `Install Python packages` button. Don't worry if Blender seems to hang and not respond for a while. You can monitor the progress of the installation in Blender's Console Window. If you are on Windows, it will be automatically opened when the aforementioned button is pressed. Otherwise, you might need to run Blender from a terminal and view the Console Window output there instead. Depending on your Internet speed, you can go get another cup of coffee while waiting.

5. Enable the Sidebar (go to `View > Sidebar` or click the `N` key) if it is not enabled yet. You can then go to `Sidebar > Solvent` and input your parameters to generate the texture that you want. After setting the parameters to your heart's content, press the `Generate Texture` button and wait. This might take a while on a CUDA-compatible GPU and even longer on a CPU. Depending on your CPU/GPU's speed and the parameters that you have chosen, you can go get yet another cup of coffee while waiting.

6. ???

7. Profit!!!

## Requirements

On top of Blender's requirements specified [here](https://www.blender.org/download/requirements/), you would also need:

- Depending on what your operating system is, around 5-6 GB of space if you use the CPU-only version of PyTorch or around 7-8 GB of space if you use the CUDA GPU version of PyTorch.
- Around 16 GB of RAM. More is highly recommended.
- A CUDA-compatible NVIDIA GPU as listed [here](https://developer.nvidia.com/cuda-gpus#compute). This is because PyTorch currently only works with CUDA-compatible NVIDIA GPUs for GPU acceleration. You could try using PyTorch with CPU (which would run more slowly) if you do not have a CUDA-compatible NVIDIA GPU. The GPU should also possess sufficient dedicated GPU memory, preferably 6.5 GB or more. Otherwise, if the GPU possess insufficient dedicated memory, Blender might intermittently and randomly crash.

## Known Issues

- As mentioned [here](https://pytorch.org/get-started/locally/), MacOS binaries of PyTorch do not support CUDA yet. Thus, building and installing PyTorch from source would be required if you would like to use CUDA with it. You can follow the instructions outlined [here](https://github.com/pytorch/pytorch#from-source) to do that and ensure that Blender's Python interpreter can detect it.

- For some reason, if you are unlucky, the PyTorch package download speed can be quite slow as mentioned [here](https://github.com/pytorch/pytorch/issues/17023). Until the root cause and a permanent solution can be found, this issue will remain.

## TODOs

- [ ] Add image upscaler module.
- [ ] Add a dropdown for the user to select different schedulers.
- [ ] Integrate the `inpaint` and `img2img` Stable Diffusion pipelines into the add-on as well (if the use case makes sense).
- [ ] Ensure cross-compatibility across Windows, Mac, and Linux.
- [ ] Check the minimum Blender version that this add-on can support.

If you encounter any problems or have any suggestions, feel free to raise an [issue](https://github.com/jamestiotio/solvent/issues) or a [pull request](https://github.com/jamestiotio/solvent/pulls)!

## Licenses

This add-on is licensed under the GNU General Public License v2.0 as attached in this repository [here](./LICENSE), the [same license](https://git.blender.org/gitweb/gitweb.cgi/blender.git/blob/HEAD:/doc/license/GPL-license.txt) that Blender uses.

This project depends on these following projects:

| Project | License |
|:-------:|:--------------------:|
| [PyTorch](https://pytorch.org/) | [3-Clause BSD License](https://github.com/pytorch/pytorch/blob/master/LICENSE) |
| [Diffusers](https://github.com/huggingface/diffusers) | [Apache License 2.0](https://github.com/huggingface/diffusers/blob/main/LICENSE) |
| [Transformers](https://github.com/huggingface/transformers) | [Apache License 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| [Stable Diffusion Pre-Trained Model Checkpoint Weights](https://huggingface.co/CompVis/stable-diffusion) | [CreativeML Open RAIL-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license) |

Any generated texture images are subject to the Creative Commons CC0 1.0 Universal Public Domain Dedication License specified [here](https://creativecommons.org/publicdomain/zero/1.0/legalcode).

Use of this add-on implies that you agree with all of the terms and conditions mentioned in all of the licenses above.
