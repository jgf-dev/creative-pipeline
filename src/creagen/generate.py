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
    """
    Generate images from a text prompt, print each resulting image URL, and save the images to disk.
    
    Parameters:
        prompt (str): Text prompt describing the desired image content.
        model (str): Model identifier to use for image generation.
        negative_prompt (str | Omit): Optional negative prompt to bias generation away from specified content; use `omit` to skip.
        guidance (float): Guidance scale controlling adherence to the prompt.
        steps (int): Number of diffusion steps to run.
        seed (int): Random seed used for generation; default is chosen at call time with random.randint(0, 1000000).
        height (int | Omit): Optional image height in pixels; use `omit` to use the model/default.
        width (int | Omit): Optional image width in pixels; use `omit` to use the model/default.
    """
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
