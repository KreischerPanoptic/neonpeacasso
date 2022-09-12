# from dataclasses import dataclass
import random
from typing import Any, List, Optional, Union

from pydantic.dataclasses import dataclass


@dataclass
class GeneratorConfig:
    """Configuration for a generation"""

    prompt: Union[str, List[str]]
    num_images: int = 1
    mode: str = "prompt"  # prompt, image, mask
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    eta: Optional[float] = 0.0
    # generator: Optional[Any] = None
    output_type: Optional[str] = "pil"
    strength: float = 0.8
    init_image: Any = None
    return_intermediates: bool = False
    turbo: Optional[bool] = True
    speed_mp: Optional[int] = 2
    n_iter: Optional[int] = 1
    skip_grid: Optional[bool] = True
    skip_save: Optional[bool] = False
    fixed_code: Optional[bool] = True
    C: Optional[int] = 4
    f: Optional[int] = 8
    device: Optional[str] = "cuda"
    unet_bs: Optional[int] = 1
    precision: str = "autocast"
    format: str = "png"
    sampler: str = "plms"
    from_file: Optional[bool] = False

    outpath = "."
    scale = guidance_scale
    ddim_steps = num_inference_steps
    ddim_eta = eta


