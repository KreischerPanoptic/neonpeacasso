import os
from dataclasses import asdict
from typing import List

import torch
from PIL import Image
from omegaconf import OmegaConf
from torch import autocast
import sys

from peacasso.neon_diff.ldm.util import instantiate_from_config

sys.path.append("neon_diff/")
sys.path.append("neon_diff/optimizedSD")
from peacasso.datamodel import GeneratorConfig
from neon_diff.optimizedSD.optimized_txt2img import get_image
# from peacasso.pipelines import StableDiffusionPipeline


# from diffusers import StableDiffusionPipeline

def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


class ImageGenerator:
    """Generate image from prompt"""

    def __init__(
        self,
        model: str = "CompVis/stable-diffusion-v1-4",
        cuda_device: int = 0,
    ) -> None:
        sd = load_model_from_config("neon_diff/models/ldm/stable-diffusion-v1/model.ckpt")
        li, lo = [], []
        for key, value in sd.items():
            sp = key.split(".")
            if (sp[0]) == "model":
                if "input_blocks" in sp:
                    li.append(key)
                elif "middle_block" in sp:
                    li.append(key)
                elif "time_embed" in sp:
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            sd["model1." + key[6:]] = sd.pop(key)
        for key in lo:
            sd["model2." + key[6:]] = sd.pop(key)

        config = OmegaConf.load("neon_diff/optimizedSD/v1-inference.yaml")

        self._model = instantiate_from_config(config.modelUNet)
        _, _ = self._model.load_state_dict(sd, strict=False)
        self._model.eval()
        self._model.unet_bs = 1
        self._model.cdevice = torch.device(cuda_device)
        self._model.turbo = "True"

        self._modelCS = instantiate_from_config(config.modelCondStage)
        _, _ = self._modelCS.load_state_dict(sd, strict=False)
        self._modelCS.eval()
        self._modelCS.cond_stage_model.device = torch.device(cuda_device)

        self._modelFS = instantiate_from_config(config.modelFirstStage)
        _, _ = self._modelFS.load_state_dict(sd, strict=False)
        self._modelFS.eval()
        del sd
        self._model.half()
        self._modelCS.half()
        self._modelFS.half()
        self._model.to(torch.device(cuda_device))
        self._modelFS.to(torch.device(cuda_device))
        self._model.to(torch.device(cuda_device))

        # assert token is not None, "HF_API_TOKEN environment variable must be set."
        # self.device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     model,
        #     revision="fp16",
        #     torch_dtype=torch.float16,
        #     use_auth_token=token,
        # ).to(self.device)

    def generate(self, config: GeneratorConfig) -> Image:
        """Generate image from prompt"""
        print(config)
        return get_image(config, self._model, self._modelCS, self._modelFS)
        # config.prompt = [config.prompt] * config.num_images
        # with autocast("cuda" if torch.cuda.is_available() else "cpu"):
        #     results = self.pipe(**asdict(config))
        # return results

    def list_cuda(self) -> List[int]:
        """List available cuda devices
        Returns:
            List[int]: List of available cuda devices
        """
        available_gpus = [i for i in range(torch.cuda.device_count())]
        return available_gpus
