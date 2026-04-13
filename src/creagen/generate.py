import random

from dotenv import load_dotenv
from rich import print
from together import Omit, Together, omit

from creagen.utils import save_image

load_dotenv()
client = Together()


def genImage(
    prompt: str,
    model: str,
    negative_prompt: str | Omit = omit,
    guidance: float = 4.0,
    steps: int = 20,
    seed: int = random.randint(0, 1000000),
    height: int | Omit = omit,
    width: int | Omit = omit,
):
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
        prompt="a very good looking 18yo football player after a game, sweaty and shirtless, wearing only his football pants",
        model="Rundiffusion/Juggernaut-Lightning-Flux",
    )
