import base64
import io
import os
import re
from typing import List

from PIL import Image


def get_dirs(path: str) -> List[str]:
    return next(os.walk(path))[1]


def base64_to_pil(base64_string: str) -> Image:
    base64_string = re.sub('^data:image/.+;base64,', '', base64_string)
    img_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_bytes))
    return img
