from together.types import ModelObject
import jinja2
import io
from PIL import Image
import requests
from together import Together
from rich import print
from together.types.model_object import Pricing
from datetime import datetime

def render_prompt_from_template(template: str, **kwargs):
	with open(template, "r") as f_template:
		template_content = f_template.read()

	rendered_prompt = jinja2.Template(template_content).render(**kwargs)
	print(rendered_prompt)

	return rendered_prompt


def save_image(image_url, filename: str = "image"):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    image_data = requests.get(image_url).content
    image = Image.open(io.BytesIO(image_data))
    image.save(f"{filename}-{now}.png")



def get_together_models():
	client = Together()
	api_list: list[ModelObject] = client.models.list()
	for x in api_list:
		if x.type != None:
			out = f"{x.id} | [bold]{x.display_name}[/bold] [blue]{x.type}[/blue]\n"
			out += f"  [yellow]{x.context_length}[/yellow]\n" if x.context_length else ""
			out += f"  [green]{x.pricing.base}[/green] [red]in:{x.pricing.input}[/red] [yellow]out:{x.pricing.output}[/yellow] [blue]{x.pricing.hourly}[/blue]\n" if x.pricing else ""
			out += f"License\t{x.license}\n" if x.license else ""
			out += f"{x.organization}\n" if x.organization else "" + f" [fine]{x.link}\n" if x.link else ""
			print(out)



if __name__ == "__main__":
    get_together_models()
