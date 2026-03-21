from typing import Literal, Union
from datetime import datetime
from creagen.utils import save_image
from dotenv import load_dotenv
from PIL import Image
import random
from rich import print
from together import omit, Together, Omit
from together.types import ImageDataB64, ImageDataURL

load_dotenv()
client = Together()




def genImage(prompt: str, model: str, negative_prompt: str|Omit = omit, guidance: float = 4.0, steps: int = 20, seed: int = random.randint(0, 1000000), height: int|Omit = omit, width: int|Omit = omit):
    response = response = client.images.generate(
        prompt=prompt,
        model=model,
        image_loras=omit,
        image_url=omit,
        guidance_scale=guidance,
        disable_safety_checker=True,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        response_format="url",
        output_format="png",
    )
    for image in response.data:
        print(f"Image URL: {image.url}")
        save_image(image.url, "image")





if __name__ == "__main__":
    genImage(
        prompt="a very good looking 18yo football player shirtless",
        model="Rundiffusion/Juggernaut-Lightning-Flux"
    )
