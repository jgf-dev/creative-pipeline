from typing import Literal, Union
from datetime import datetime

import dotenv
import jinja2
from PIL import Image
from rich import print
from together import omit, Together, Omit

dotenv.load_dotenv()
client = Together()

api_list = client.models.list()
print(x for x in api_list)


def genImage(prompt: str, model: str, format: Omit|Literal["base64", "url"]):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")



    response = client.images.generate(
        prompt=prompt,
        image_url=omit,
        model=model,
        image_loras=[],
        disable_safety_checker=True,
        steps=20,
        seed=42,
        response_format=format,
        guidance_scale=omit,
        negative_prompt=omit,
        height=omit,
    )
    print(response)

    image_bytes = response.data[0].b64_json
    print(image_bytes)
    image = Image.open(image_bytes)

    image.save(f"image-{now.strftime('%Y%m%d-%H%M%S')}.png")
